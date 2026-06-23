"""
transfer_build_tli_departure_state_monte_mpi.py

Purpose
-------
MPI-only ballistic TLI-like / cislunar release-state sweep.

This script is the parallel version of transfer_build_tli_departure_state_monte.py.
It splits release-state samples across MPI ranks using round-robin assignment,
propagates each assigned case ballistically with the n-body propagator, gathers
the summary records on rank 0, and writes final CSV/JSON products.

This script intentionally does not generate plots. Use the serial/non-MPI script
for interactive visualization or single-process debugging.

Run examples
------------
    mpiexec -n 8 python transfer_build_tli_departure_state_monte_mpi.py transfer_tli_departure_config_monte_mpi.yaml
    srun -n 192 python transfer_build_tli_departure_state_monte_mpi.py transfer_tli_departure_config_monte_mpi.yaml

State convention
----------------
All states are Earth-centered J2000/EME:
    [x, y, z, vx, vy, vz]

position [km], velocity [km/s].
"""

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

import argparse
import json
from pathlib import Path
from itertools import product

import yaml
import numpy as np
import pandas as pd
import spiceypy as spice

from n_body_integrator import NBodyPropagator


MU_EARTH_KM3_S2 = 398600.4418


RESULT_KEYS = [
    "proxy_cost",
    "final_position_error_norm_km",
    "final_velocity_error_norm_kms",
    "final_velocity_error_norm_mps",
    "min_geocentric_distance_km",
    "max_geocentric_distance_km",
    "final_geocentric_distance_km",
    "min_moon_distance_km",
    "final_moon_distance_km",
    "initial_C3_km2_s2",
    "final_C3_km2_s2",
    "final_x_geo_j2000_km",
    "final_y_geo_j2000_km",
    "final_z_geo_j2000_km",
    "final_vx_geo_j2000_kms",
    "final_vy_geo_j2000_kms",
    "final_vz_geo_j2000_kms",
    "initial_x_geo_j2000_km",
    "initial_y_geo_j2000_km",
    "initial_z_geo_j2000_km",
    "initial_vx_geo_j2000_kms",
    "initial_vy_geo_j2000_kms",
    "initial_vz_geo_j2000_kms",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI-only ballistic TLI-like departure-state sweep."
    )
    parser.add_argument(
        "config",
        help="Path to the MPI YAML config file.",
    )
    return parser.parse_args()


def rank_print(rank, message):
    print(f"[rank {rank:04d}] {message}", flush=True)

# =============================================================================
# Basic utilities
# =============================================================================

def unit(vec):
    vec = np.asarray(vec, dtype=float)
    n = np.linalg.norm(vec)
    if n <= 0.0:
        raise ValueError("Cannot normalize zero vector.")
    return vec / n


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_spice_kernels(config):
    for kernel in config["spice"]["kernels"]:
        spice.furnsh(kernel)


def utc_to_jdtdb(utc_string):
    et = spice.str2et(str(utc_string))
    return float(spice.unitim(et, "ET", "JDTDB"))


def jdtdb_to_et(jdtdb):
    return float(spice.unitim(float(jdtdb), "JDTDB", "ET"))


def jdtdb_to_utc(jdtdb):
    et = jdtdb_to_et(jdtdb)
    return spice.et2utc(float(et), "ISOC", 6)


def compute_C3_geo(state):
    state = np.asarray(state, dtype=float)
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])
    return v**2 - 2.0 * MU_EARTH_KM3_S2 / r


def speed_from_C3(C3_km2_s2, r_mag_km):
    C3 = float(C3_km2_s2)
    r_mag = float(r_mag_km)

    speed_sq = C3 + 2.0 * MU_EARTH_KM3_S2 / r_mag

    if speed_sq <= 0.0:
        raise ValueError(
            f"Invalid C3/r combination: C3={C3}, r={r_mag}, v^2={speed_sq}"
        )

    return np.sqrt(speed_sq)


# =============================================================================
# LPF target
# =============================================================================

def load_lpf_target_state(config):
    lpf_path = config["files"]["lpf_orbit_csv"]
    target_cfg = config["target"]

    df = pd.read_csv(lpf_path)

    idx = int(target_cfg["target_index"])
    time_col = target_cfg["time_column"]
    cols = target_cfg["state_columns_geo_eme"]

    if idx < 0 or idx >= len(df):
        raise IndexError(f"target_index={idx} outside LPF file length {len(df)}.")

    missing = [c for c in cols + [time_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing LPF columns: {missing}")

    row = df.iloc[idx]
    target_time_utc = str(row[time_col])
    target_state = row[cols].to_numpy(dtype=float)

    return target_time_utc, target_state, row


# =============================================================================
# Earth-Moon basis
# =============================================================================

def get_moon_state_geo_j2000(t_jdtdb):
    et = jdtdb_to_et(t_jdtdb)

    moon_state, _ = spice.spkgeo(
        301,        # Moon
        et,
        "J2000",
        399,        # Earth
    )

    moon_state = np.asarray(moon_state, dtype=float)
    return moon_state[:3], moon_state[3:]


def get_earth_moon_basis(t_jdtdb):
    r_moon, v_moon = get_moon_state_geo_j2000(t_jdtdb)

    e1 = unit(r_moon)
    e3 = unit(np.cross(r_moon, v_moon))
    e2 = unit(np.cross(e3, e1))

    return e1, e2, e3, r_moon, v_moon


def get_body_geo_j2000_position(body_id, t_jdtdb):
    et = jdtdb_to_et(t_jdtdb)
    r_body, _ = spice.spkpos(str(body_id), et, "J2000", "NONE", "399")
    return np.asarray(r_body, dtype=float)


# =============================================================================
# State construction
# =============================================================================

def build_tli_like_state_from_sample(sample, basis):
    """
    Build TLI-like release state from one sampled parameter set.

    sample keys:
        r0_mag_km
        C3_km2_s2
        phi_r_deg
        phi_v_deg
        out_of_plane_r_deg
        out_of_plane_v_deg
    """

    e1, e2, e3, r_moon, v_moon = basis

    r0_mag = float(sample["r0_mag_km"])
    C3 = float(sample["C3_km2_s2"])

    phi_r = np.deg2rad(float(sample["phi_r_deg"]))
    phi_v = np.deg2rad(float(sample["phi_v_deg"]))
    out_r = np.deg2rad(float(sample["out_of_plane_r_deg"]))
    out_v = np.deg2rad(float(sample["out_of_plane_v_deg"]))

    r_hat_plane = np.cos(phi_r) * e1 + np.sin(phi_r) * e2
    r_hat = np.cos(out_r) * r_hat_plane + np.sin(out_r) * e3
    r_hat = unit(r_hat)

    v_hat_plane = np.cos(phi_v) * e1 + np.sin(phi_v) * e2
    v_hat = np.cos(out_v) * v_hat_plane + np.sin(out_v) * e3
    v_hat = unit(v_hat)

    v0_mag = speed_from_C3(C3, r0_mag)

    r0 = r0_mag * r_hat
    v0 = v0_mag * v_hat

    return np.hstack([r0, v0])


# =============================================================================
# Sampling
# =============================================================================

def linspace_from_config(block):
    return np.linspace(float(block["min"]), float(block["max"]), int(block["n"]))


def generate_grid_samples(config):
    sweep = config["sweep"]

    r0_vals = linspace_from_config(sweep["r0_mag_km"])
    C3_vals = linspace_from_config(sweep["C3_km2_s2"])
    phi_r_vals = linspace_from_config(sweep["phi_r_deg"])
    phi_v_vals = linspace_from_config(sweep["phi_v_deg"])
    out_r_vals = linspace_from_config(sweep["out_of_plane_r_deg"])
    out_v_vals = linspace_from_config(sweep["out_of_plane_v_deg"])

    samples = []

    for k, values in enumerate(product(
        r0_vals,
        C3_vals,
        phi_r_vals,
        phi_v_vals,
        out_r_vals,
        out_v_vals,
    )):
        r0, C3, phi_r, phi_v, out_r, out_v = values

        samples.append({
            "sample_id": k,
            "r0_mag_km": float(r0),
            "C3_km2_s2": float(C3),
            "phi_r_deg": float(phi_r),
            "phi_v_deg": float(phi_v),
            "out_of_plane_r_deg": float(out_r),
            "out_of_plane_v_deg": float(out_v),
        })

    return samples


def random_uniform_from_config(rng, block, n):
    return rng.uniform(float(block["min"]), float(block["max"]), int(n))


def generate_random_samples(config):
    sweep = config["sweep"]
    n = int(sweep["n_random"])
    rng = np.random.default_rng(int(sweep.get("random_seed", 1)))

    r0_vals = random_uniform_from_config(rng, sweep["r0_mag_km"], n)
    C3_vals = random_uniform_from_config(rng, sweep["C3_km2_s2"], n)
    phi_r_vals = random_uniform_from_config(rng, sweep["phi_r_deg"], n)
    phi_v_vals = random_uniform_from_config(rng, sweep["phi_v_deg"], n)
    out_r_vals = random_uniform_from_config(rng, sweep["out_of_plane_r_deg"], n)
    out_v_vals = random_uniform_from_config(rng, sweep["out_of_plane_v_deg"], n)

    samples = []

    for k in range(n):
        samples.append({
            "sample_id": k,
            "r0_mag_km": float(r0_vals[k]),
            "C3_km2_s2": float(C3_vals[k]),
            "phi_r_deg": float(phi_r_vals[k]),
            "phi_v_deg": float(phi_v_vals[k]),
            "out_of_plane_r_deg": float(out_r_vals[k]),
            "out_of_plane_v_deg": float(out_v_vals[k]),
        })

    return samples


def generate_samples(config):
    mode = config["sweep"].get("mode", "grid").lower()

    if mode == "grid":
        return generate_grid_samples(config)

    if mode == "random":
        return generate_random_samples(config)

    raise ValueError(f"Unknown sweep mode: {mode}")


# =============================================================================
# Propagation
# =============================================================================

def load_nbody_constants(config):
    with open(config["files"]["constants_yaml"], "r") as f:
        return yaml.safe_load(f)


def build_nbody_propagator(config):
    constants = load_nbody_constants(config)
    prop_cfg = config["propagation"]

    return NBodyPropagator(
        spice=spice,
        config=constants,
        bodies=tuple(prop_cfg["bodies"]),
        frame=prop_cfg.get("frame", "J2000"),
        origin=prop_cfg.get("origin", "399"),
        rtol=float(prop_cfg.get("rtol", 1e-10)),
        atol=float(prop_cfg.get("atol", 1e-12)),
        method=prop_cfg.get("method", "DOP853"),
    )


def build_time_grid(t0_jdtdb, tf_jdtdb, step_days):
    t0 = float(t0_jdtdb)
    tf = float(tf_jdtdb)
    step = float(step_days)

    if step <= 0:
        raise ValueError("step_days must be positive.")

    if tf < t0:
        raise ValueError("Final epoch must be after initial epoch.")

    t_grid = np.arange(t0, tf, step)

    if t_grid.size == 0 or not np.isclose(t_grid[0], t0):
        t_grid = np.insert(t_grid, 0, t0)

    if t_grid[-1] < tf:
        t_grid = np.append(t_grid, tf)

    return t_grid


def propagate_sample(propagator, state0, t_depart_jdtdb, t_target_jdtdb, config):
    prop_cfg = config["propagation"]

    debug_duration = prop_cfg.get("debug_duration_days", None)
    if debug_duration is not None:
        tf = float(t_depart_jdtdb) + float(debug_duration)
    else:
        tf = float(t_target_jdtdb)

    t_grid = build_time_grid(
        t0_jdtdb=t_depart_jdtdb,
        tf_jdtdb=tf,
        step_days=float(prop_cfg["step_days"]),
    )

    X = propagator.propagate(
        x0_km=state0,
        t0_jdtdb=t_depart_jdtdb,
        t1_jdtdb=t_grid,
    )

    return t_grid, X


# =============================================================================
# Metrics and cost
# =============================================================================

def compute_moon_distance_history(t_grid, X):
    distances = np.zeros(len(t_grid))

    for i, t in enumerate(t_grid):
        r_moon = get_body_geo_j2000_position("301", t)
        distances[i] = np.linalg.norm(X[i, :3] - r_moon)

    return distances


def compute_proxy_cost(metrics, sample, config):
    """
    Proxy cost used for ranking ballistic cases.

    This is NOT the final maneuver ΔV.
    It approximates the future shooting objective:

        position miss + velocity miss + plausibility penalties
    """

    cost_cfg = config["cost"]

    r_scale = float(cost_cfg["r_error_scale_km"])
    v_scale = float(cost_cfg["v_error_scale_kms"])

    J = 0.0

    J += metrics["final_position_error_norm_km"] / r_scale
    J += metrics["final_velocity_error_norm_kms"] / v_scale

    C3_weight = float(cost_cfg.get("C3_weight", 0.0))
    C3_ref = float(cost_cfg.get("C3_reference_km2_s2", sample["C3_km2_s2"]))
    J += C3_weight * abs(float(sample["C3_km2_s2"]) - C3_ref)

    moon_soft = float(cost_cfg.get("min_moon_distance_soft_km", 0.0))
    moon_w = float(cost_cfg.get("moon_close_penalty_weight", 0.0))

    if moon_soft > 0 and metrics["min_moon_distance_km"] < moon_soft:
        J += moon_w * (moon_soft - metrics["min_moon_distance_km"]) / moon_soft

    earth_soft = float(cost_cfg.get("min_earth_distance_soft_km", 0.0))
    earth_w = float(cost_cfg.get("earth_close_penalty_weight", 0.0))

    if earth_soft > 0 and metrics["min_geocentric_distance_km"] < earth_soft:
        J += earth_w * (earth_soft - metrics["min_geocentric_distance_km"]) / earth_soft

    return float(J)


def evaluate_trajectory(sample, t_grid, X, target_state, config):
    r = X[:, :3]
    v = X[:, 3:]

    r_norm = np.linalg.norm(r, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    C3_hist = v_norm**2 - 2.0 * MU_EARTH_KM3_S2 / r_norm

    final_state = X[-1]
    dr_f = final_state[:3] - target_state[:3]
    dv_f = final_state[3:] - target_state[3:]

    moon_dist = compute_moon_distance_history(t_grid, X)

    metrics = {
        "final_position_error_norm_km": float(np.linalg.norm(dr_f)),
        "final_velocity_error_norm_kms": float(np.linalg.norm(dv_f)),
        "final_velocity_error_norm_mps": float(1000.0 * np.linalg.norm(dv_f)),

        "min_geocentric_distance_km": float(np.min(r_norm)),
        "max_geocentric_distance_km": float(np.max(r_norm)),
        "final_geocentric_distance_km": float(r_norm[-1]),

        "min_moon_distance_km": float(np.min(moon_dist)),
        "final_moon_distance_km": float(moon_dist[-1]),

        "initial_C3_km2_s2": float(C3_hist[0]),
        "final_C3_km2_s2": float(C3_hist[-1]),

        "final_x_geo_j2000_km": float(final_state[0]),
        "final_y_geo_j2000_km": float(final_state[1]),
        "final_z_geo_j2000_km": float(final_state[2]),
        "final_vx_geo_j2000_kms": float(final_state[3]),
        "final_vy_geo_j2000_kms": float(final_state[4]),
        "final_vz_geo_j2000_kms": float(final_state[5]),

        "initial_x_geo_j2000_km": float(X[0, 0]),
        "initial_y_geo_j2000_km": float(X[0, 1]),
        "initial_z_geo_j2000_km": float(X[0, 2]),
        "initial_vx_geo_j2000_kms": float(X[0, 3]),
        "initial_vy_geo_j2000_kms": float(X[0, 4]),
        "initial_vz_geo_j2000_kms": float(X[0, 5]),
    }

    metrics["proxy_cost"] = compute_proxy_cost(metrics, sample, config)

    return metrics


# =============================================================================
# Save helpers
# =============================================================================

def trajectory_to_dataframe(t_grid, X, t_depart_jdtdb):
    r = X[:, :3]
    v = X[:, 3:]

    r_norm = np.linalg.norm(r, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    C3 = v_norm**2 - 2.0 * MU_EARTH_KM3_S2 / r_norm

    return pd.DataFrame({
        "jdtdb": t_grid,
        "utc": [jdtdb_to_utc(t) for t in t_grid],
        "t_days_since_departure": t_grid - t_depart_jdtdb,

        "x_geo_j2000_km": X[:, 0],
        "y_geo_j2000_km": X[:, 1],
        "z_geo_j2000_km": X[:, 2],
        "vx_geo_j2000_kms": X[:, 3],
        "vy_geo_j2000_kms": X[:, 4],
        "vz_geo_j2000_kms": X[:, 5],

        "r_geo_km": r_norm,
        "v_geo_kms": v_norm,
        "C3_geo_km2_s2": C3,
    })


def save_sweep_outputs(
    config,
    summary_df,
    best_record,
    average_record,
    best_traj,
    average_traj,
    t_depart_jdtdb,
    t_target_jdtdb,
    target_state,
):
    output_dir = Path(config["files"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_cfg = config.get("output", {})

    summary_csv = output_dir / "ballistic_release_sweep_summary.csv"
    if bool(output_cfg.get("save_summary_csv", True)):
        summary_df.to_csv(summary_csv, index=False)

    summary_json = output_dir / "ballistic_release_sweep_metadata.json"

    metadata = {
        "n_cases_total": int(len(summary_df)),
        "n_cases_success": int((summary_df["status"] == "OK").sum()),
        "n_cases_failed": int((summary_df["status"] != "OK").sum()),

        "target_index": int(config["target"]["target_index"]),
        "departure_time_jdtdb": float(t_depart_jdtdb),
        "departure_time_utc": jdtdb_to_utc(t_depart_jdtdb),
        "target_time_jdtdb": float(t_target_jdtdb),
        "target_time_utc": jdtdb_to_utc(t_target_jdtdb),

        "target_state_geo_eme_km_kms": target_state.tolist(),

        "best_sample_id": int(best_record["sample_id"]) if best_record is not None else None,
        "best_proxy_cost": float(best_record["proxy_cost"]) if best_record is not None else None,

        "average_sample_id": int(average_record["sample_id"]) if average_record is not None else None,
        "average_proxy_cost": float(average_record["proxy_cost"]) if average_record is not None else None,
    }

    if bool(output_cfg.get("save_summary_json", True)):
        with open(summary_json, "w") as f:
            json.dump(metadata, f, indent=2)

    if best_traj is not None and bool(output_cfg.get("save_best_trajectory_csv", True)):
        t_grid_best, X_best = best_traj
        df_best = trajectory_to_dataframe(t_grid_best, X_best, t_depart_jdtdb)
        df_best.to_csv(output_dir / "ballistic_best_proxy_cost_trajectory.csv", index=False)

    if average_traj is not None and bool(output_cfg.get("save_average_trajectory_csv", True)):
        t_grid_avg, X_avg = average_traj
        df_avg = trajectory_to_dataframe(t_grid_avg, X_avg, t_depart_jdtdb)
        df_avg.to_csv(output_dir / "ballistic_average_proxy_cost_trajectory.csv", index=False)

    return summary_csv, summary_json


# =============================================================================
# Selection helpers
# =============================================================================

def select_best_and_average(summary_df):
    ok = summary_df[summary_df["status"] == "OK"].copy()

    if ok.empty:
        return None, None

    ok = ok.sort_values("proxy_cost").reset_index(drop=True)

    best_record = ok.iloc[0].to_dict()

    median_cost = float(ok["proxy_cost"].median())
    ok["median_cost_distance"] = np.abs(ok["proxy_cost"] - median_cost)

    average_record = ok.sort_values("median_cost_distance").iloc[0].to_dict()

    return best_record, average_record




# =============================================================================
# MPI helpers
# =============================================================================

def split_samples_round_robin(samples, rank, size):
    """Deterministic round-robin assignment of global samples to each MPI rank."""
    return samples[rank::size]


def run_one_sample(
    sample,
    basis,
    propagator,
    t_depart_jdtdb,
    t_target_jdtdb,
    target_state,
    config,
):
    """Propagate and evaluate one release-state sample, returning one CSV record."""
    record = dict(sample)

    try:
        state0 = build_tli_like_state_from_sample(sample, basis)

        t_grid, X = propagate_sample(
            propagator=propagator,
            state0=state0,
            t_depart_jdtdb=t_depart_jdtdb,
            t_target_jdtdb=t_target_jdtdb,
            config=config,
        )

        metrics = evaluate_trajectory(
            sample=sample,
            t_grid=t_grid,
            X=X,
            target_state=target_state,
            config=config,
        )

        record.update(metrics)
        record["status"] = "OK"
        record["error_message"] = ""

    except Exception as e:
        record["status"] = "FAILED"
        record["error_message"] = str(e)

        # Fill key fields with NaN so CSV structure remains stable.
        for key in RESULT_KEYS:
            record[key] = np.nan

    return record


def sample_from_record(record):
    """Reconstruct a sample dictionary from a summary row/record."""
    return {
        "sample_id": int(record["sample_id"]),
        "r0_mag_km": float(record["r0_mag_km"]),
        "C3_km2_s2": float(record["C3_km2_s2"]),
        "phi_r_deg": float(record["phi_r_deg"]),
        "phi_v_deg": float(record["phi_v_deg"]),
        "out_of_plane_r_deg": float(record["out_of_plane_r_deg"]),
        "out_of_plane_v_deg": float(record["out_of_plane_v_deg"]),
    }


def repropagate_record(record, basis, propagator, t_depart_jdtdb, t_target_jdtdb, config):
    """Repropagate a selected best/median record for trajectory CSV output."""
    if record is None:
        return None

    sample = sample_from_record(record)
    state0 = build_tli_like_state_from_sample(sample, basis)

    return propagate_sample(
        propagator=propagator,
        state0=state0,
        t_depart_jdtdb=t_depart_jdtdb,
        t_target_jdtdb=t_target_jdtdb,
        config=config,
    )


def save_rank_partial_outputs(config, local_records, rank):
    """Optionally save one partial CSV per rank for debugging/recovery."""
    mpi_cfg = config.get("mpi", {})

    if not bool(mpi_cfg.get("save_rank_outputs", False)):
        return None

    output_dir = Path(config["files"]["output_dir"])
    rank_dir = output_dir / str(mpi_cfg.get("rank_output_subdir", "rank_outputs"))
    rank_dir.mkdir(parents=True, exist_ok=True)

    path = rank_dir / f"ballistic_rank_{rank:04d}.csv"
    pd.DataFrame(local_records).to_csv(path, index=False)
    return path


# =============================================================================
# Main MPI sweep
# =============================================================================

def main():
    args = parse_args()
    config = load_config(args.config)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    output_dir = Path(config["files"]["output_dir"])

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Make sure output_dir exists before any optional per-rank files are written.
    comm.Barrier()

    load_spice_kernels(config)

    target_time_utc, target_state, target_row = load_lpf_target_state(config)

    tof_days = float(config["epoch"]["tof_days"])
    t_target_jdtdb = utc_to_jdtdb(target_time_utc)
    t_depart_jdtdb = t_target_jdtdb - tof_days

    basis = get_earth_moon_basis(t_depart_jdtdb)

    samples = generate_samples(config)
    local_samples = split_samples_round_robin(samples, rank, size)

    if rank == 0:
        print("MPI ballistic release sweep")
        print("---------------------------")
        print(f"Config:          {args.config}")
        print(f"MPI ranks:       {size}")
        print(f"Total samples:   {len(samples)}")
        print(f"Target index:    {config['target']['target_index']}")
        print(f"Target UTC:      {jdtdb_to_utc(t_target_jdtdb)}")
        print(f"Departure UTC:   {jdtdb_to_utc(t_depart_jdtdb)}")
        print(f"TOF days:        {tof_days:.6f}")
        print(f"Output dir:      {output_dir}")
        print()

    rank_print(rank, f"assigned {len(local_samples)} samples")

    propagator = build_nbody_propagator(config)

    mpi_cfg = config.get("mpi", {})
    progress_every = int(mpi_cfg.get("progress_every", 25))
    rank_progress = bool(mpi_cfg.get("rank_progress", True))

    local_records = []

    for i_local, sample in enumerate(local_samples):
        if rank_progress and progress_every > 0:
            if i_local == 0 or (i_local + 1) % progress_every == 0:
                rank_print(
                    rank,
                    (
                        f"running local sample {i_local + 1}/{len(local_samples)} "
                        f"(global sample_id={sample['sample_id']})"
                    ),
                )

        record = run_one_sample(
            sample=sample,
            basis=basis,
            propagator=propagator,
            t_depart_jdtdb=t_depart_jdtdb,
            t_target_jdtdb=t_target_jdtdb,
            target_state=target_state,
            config=config,
        )

        local_records.append(record)

    rank_path = save_rank_partial_outputs(config, local_records, rank)
    if rank_path is not None:
        rank_print(rank, f"wrote partial CSV: {rank_path}")

    local_ok = sum(1 for r in local_records if r.get("status") == "OK")
    local_failed = len(local_records) - local_ok
    rank_print(rank, f"complete: OK={local_ok}, FAILED={local_failed}")

    gathered_records = comm.gather(local_records, root=0)

    if rank != 0:
        return

    records = [record for rank_records in gathered_records for record in rank_records]
    summary_df = pd.DataFrame(records)

    if not summary_df.empty and "sample_id" in summary_df.columns:
        summary_df = summary_df.sort_values("sample_id").reset_index(drop=True)

    best_record, average_record = select_best_and_average(summary_df)

    best_traj = repropagate_record(
        record=best_record,
        basis=basis,
        propagator=propagator,
        t_depart_jdtdb=t_depart_jdtdb,
        t_target_jdtdb=t_target_jdtdb,
        config=config,
    )

    average_traj = repropagate_record(
        record=average_record,
        basis=basis,
        propagator=propagator,
        t_depart_jdtdb=t_depart_jdtdb,
        t_target_jdtdb=t_target_jdtdb,
        config=config,
    )

    summary_csv, metadata_json = save_sweep_outputs(
        config=config,
        summary_df=summary_df,
        best_record=best_record,
        average_record=average_record,
        best_traj=best_traj,
        average_traj=average_traj,
        t_depart_jdtdb=t_depart_jdtdb,
        t_target_jdtdb=t_target_jdtdb,
        target_state=target_state,
    )

    n_ok = int((summary_df["status"] == "OK").sum()) if not summary_df.empty else 0
    n_fail = int((summary_df["status"] != "OK").sum()) if not summary_df.empty else 0

    print()
    print("MPI sweep complete")
    print("------------------")
    print(f"Summary CSV:      {summary_csv}")
    print(f"Metadata JSON:    {metadata_json}")
    print(f"Successful cases: {n_ok}")
    print(f"Failed cases:     {n_fail}")

    if best_record is not None:
        print()
        print("Best proxy-cost case")
        print("--------------------")
        print(f"sample_id:     {int(best_record['sample_id'])}")
        print(f"proxy_cost:    {best_record['proxy_cost']:.6e}")
        print(f"r0_mag_km:     {best_record['r0_mag_km']:.3f}")
        print(f"C3:            {best_record['C3_km2_s2']:.3f}")
        print(f"phi_r_deg:     {best_record['phi_r_deg']:.3f}")
        print(f"phi_v_deg:     {best_record['phi_v_deg']:.3f}")
        print(f"pos err [km]:  {best_record['final_position_error_norm_km']:.6e}")
        print(f"vel err [m/s]: {best_record['final_velocity_error_norm_mps']:.6e}")

    if average_record is not None:
        print()
        print("Average/median proxy-cost case")
        print("------------------------------")
        print(f"sample_id:     {int(average_record['sample_id'])}")
        print(f"proxy_cost:    {average_record['proxy_cost']:.6e}")


if __name__ == "__main__":
    main()
