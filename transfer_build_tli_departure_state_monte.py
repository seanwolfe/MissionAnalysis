"""
ballistic_release_sweep.py

Purpose
-------
Sample a population of plausible TLI-like / cislunar rideshare release states,
propagate each ballistically with the n-body propagator, and rank them using a
proxy cost that approximates the maneuver-correction objective to be used later.

This script does NOT optimize MCC or insertion maneuvers yet.

It does:
    1. Load LPF target state at target index.
    2. Compute departure epoch = target epoch - fixed TOF.
    3. Generate a grid/random population of release states.
    4. Build each TLI-like state using the Earth-Moon basis at departure.
    5. Propagate each state ballistically to target epoch.
    6. Compute final miss, final velocity error, Moon/Earth distance metrics.
    7. Compute a proxy cost for later maneuver-correction usefulness.
    8. Save sweep summary.
    9. Optionally visualize endpoints and representative trajectories.

State convention
----------------
All states are Earth-centered J2000/EME:
    [x, y, z, vx, vy, vz]

position [km], velocity [km/s].
"""

import json
from pathlib import Path
from itertools import product

import yaml
import numpy as np
import pandas as pd
import spiceypy as spice
import matplotlib.pyplot as plt

from n_body_integrator import NBodyPropagator


MU_EARTH_KM3_S2 = 398600.4418


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
# Visualization
# =============================================================================

def set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range / 2.0, x_mid + max_range / 2.0])
    ax.set_ylim3d([y_mid - max_range / 2.0, y_mid + max_range / 2.0])
    ax.set_zlim3d([z_mid - max_range / 2.0, z_mid + max_range / 2.0])


def plot_sweep_endpoints(config, summary_df, target_state, best_record, average_record):
    viz = config.get("visualization", {}).get("sweep_endpoints", {})
    if not bool(viz.get("enabled", False)):
        return

    ok = summary_df[summary_df["status"] == "OK"].copy()
    if ok.empty:
        return

    output_dir = Path(config["files"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        ok["final_x_geo_j2000_km"],
        ok["final_y_geo_j2000_km"],
        ok["final_z_geo_j2000_km"],
        c=ok["proxy_cost"],
        s=float(viz.get("marker_size", 15)),
        alpha=0.65,
        label="Ballistic endpoints",
    )

    ax.scatter([0.0], [0.0], [0.0], s=80, label="Earth")

    ax.scatter(
        [target_state[0]],
        [target_state[1]],
        [target_state[2]],
        marker="x",
        s=90,
        label="LPF target",
    )

    if best_record is not None:
        ax.scatter(
            [best_record["final_x_geo_j2000_km"]],
            [best_record["final_y_geo_j2000_km"]],
            [best_record["final_z_geo_j2000_km"]],
            marker="*",
            s=float(viz.get("best_marker_size", 90)),
            label="Best proxy-cost case",
        )

    if average_record is not None:
        ax.scatter(
            [average_record["final_x_geo_j2000_km"]],
            [average_record["final_y_geo_j2000_km"]],
            [average_record["final_z_geo_j2000_km"]],
            marker="D",
            s=float(viz.get("average_marker_size", 90)),
            label="Average proxy-cost case",
        )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label("Proxy cost")

    all_points = np.vstack([
        ok[[
            "final_x_geo_j2000_km",
            "final_y_geo_j2000_km",
            "final_z_geo_j2000_km",
        ]].to_numpy(),
        target_state[:3].reshape(1, 3),
        np.zeros((1, 3)),
    ])

    max_abs = np.max(np.abs(all_points))
    lim = 1.1 * max_abs

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    ax.set_xlabel("GEO J2000 X [km]")
    ax.set_ylabel("GEO J2000 Y [km]")
    ax.set_zlabel("GEO J2000 Z [km]")

    ax.set_title("Ballistic sweep final endpoints")
    ax.legend(fontsize=8)

    set_axes_equal_3d(ax)
    plt.tight_layout()

    if bool(viz.get("save_plot", True)):
        filename = viz.get("filename", "ballistic_sweep_endpoints.png")
        fig.savefig(output_dir / filename, dpi=300)

    if bool(viz.get("show_plot", False)):
        plt.show()

    plt.close(fig)


def plot_representative_trajectories(
    config,
    best_traj,
    average_traj,
    target_state,
    best_record,
    average_record,
):
    viz = config.get("visualization", {}).get("representative_trajectories", {})
    if not bool(viz.get("enabled", False)):
        return

    output_dir = Path(config["files"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    all_points = [target_state[:3].reshape(1, 3), np.zeros((1, 3))]

    if best_traj is not None and bool(viz.get("include_best", True)):
        t_best, X_best = best_traj
        r_best = X_best[:, :3]
        all_points.append(r_best)

        label = "Best proxy-cost trajectory"
        if best_record is not None:
            label += f" (id={int(best_record['sample_id'])})"

        ax.plot(r_best[:, 0], r_best[:, 1], r_best[:, 2], label=label)
        ax.scatter([r_best[0, 0]], [r_best[0, 1]], [r_best[0, 2]], marker="o", s=50)
        ax.scatter([r_best[-1, 0]], [r_best[-1, 1]], [r_best[-1, 2]], marker="*", s=80)

    if average_traj is not None and bool(viz.get("include_average", True)):
        t_avg, X_avg = average_traj
        r_avg = X_avg[:, :3]
        all_points.append(r_avg)

        label = "Average proxy-cost trajectory"
        if average_record is not None:
            label += f" (id={int(average_record['sample_id'])})"

        ax.plot(r_avg[:, 0], r_avg[:, 1], r_avg[:, 2], label=label)
        ax.scatter([r_avg[0, 0]], [r_avg[0, 1]], [r_avg[0, 2]], marker="o", s=50)
        ax.scatter([r_avg[-1, 0]], [r_avg[-1, 1]], [r_avg[-1, 2]], marker="D", s=70)

    ax.scatter([0.0], [0.0], [0.0], s=80, label="Earth")

    ax.scatter(
        [target_state[0]],
        [target_state[1]],
        [target_state[2]],
        marker="x",
        s=90,
        label="LPF target",
    )

    pts = np.vstack(all_points)
    max_abs = np.max(np.abs(pts))
    lim = 1.1 * max_abs

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    ax.set_xlabel("GEO J2000 X [km]")
    ax.set_ylabel("GEO J2000 Y [km]")
    ax.set_zlabel("GEO J2000 Z [km]")
    ax.set_title("Representative ballistic trajectories")
    ax.legend(fontsize=8)

    set_axes_equal_3d(ax)
    plt.tight_layout()

    if bool(viz.get("save_plot", True)):
        filename = viz.get("filename", "ballistic_sweep_representative_trajectories.png")
        fig.savefig(output_dir / filename, dpi=300)

    if bool(viz.get("show_plot", False)):
        plt.show()

    plt.close(fig)


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
# Main sweep
# =============================================================================

def main():
    config_path = "transfer_tli_departure_config_monte.yaml"
    config = load_config(config_path)

    output_dir = Path(config["files"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    load_spice_kernels(config)

    target_time_utc, target_state, target_row = load_lpf_target_state(config)

    tof_days = float(config["epoch"]["tof_days"])
    t_target_jdtdb = utc_to_jdtdb(target_time_utc)
    t_depart_jdtdb = t_target_jdtdb - tof_days

    print("Ballistic release sweep")
    print("-----------------------")
    print(f"Target index:     {config['target']['target_index']}")
    print(f"Target UTC:       {jdtdb_to_utc(t_target_jdtdb)}")
    print(f"Departure UTC:    {jdtdb_to_utc(t_depart_jdtdb)}")
    print(f"TOF days:         {tof_days:.6f}")
    print()

    basis = get_earth_moon_basis(t_depart_jdtdb)

    samples = generate_samples(config)
    print(f"Generated {len(samples)} samples.")

    propagator = build_nbody_propagator(config)

    records = []

    # Store only best/average trajectories after selection.
    # To avoid memory blow-up, we first save states/results only, then repropagate
    # selected samples at the end.
    for i, sample in enumerate(samples):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"Running sample {i + 1}/{len(samples)}")

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
            for key in [
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
            ]:
                record[key] = np.nan

        records.append(record)

    summary_df = pd.DataFrame(records)

    best_record, average_record = select_best_and_average(summary_df)

    best_traj = None
    average_traj = None

    # Repropagate selected samples to save representative trajectories.
    if best_record is not None:
        sample_best = {
            "sample_id": int(best_record["sample_id"]),
            "r0_mag_km": float(best_record["r0_mag_km"]),
            "C3_km2_s2": float(best_record["C3_km2_s2"]),
            "phi_r_deg": float(best_record["phi_r_deg"]),
            "phi_v_deg": float(best_record["phi_v_deg"]),
            "out_of_plane_r_deg": float(best_record["out_of_plane_r_deg"]),
            "out_of_plane_v_deg": float(best_record["out_of_plane_v_deg"]),
        }
        state_best = build_tli_like_state_from_sample(sample_best, basis)
        best_traj = propagate_sample(
            propagator=propagator,
            state0=state_best,
            t_depart_jdtdb=t_depart_jdtdb,
            t_target_jdtdb=t_target_jdtdb,
            config=config,
        )

    if average_record is not None:
        sample_avg = {
            "sample_id": int(average_record["sample_id"]),
            "r0_mag_km": float(average_record["r0_mag_km"]),
            "C3_km2_s2": float(average_record["C3_km2_s2"]),
            "phi_r_deg": float(average_record["phi_r_deg"]),
            "phi_v_deg": float(average_record["phi_v_deg"]),
            "out_of_plane_r_deg": float(average_record["out_of_plane_r_deg"]),
            "out_of_plane_v_deg": float(average_record["out_of_plane_v_deg"]),
        }
        state_avg = build_tli_like_state_from_sample(sample_avg, basis)
        average_traj = propagate_sample(
            propagator=propagator,
            state0=state_avg,
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

    plot_sweep_endpoints(
        config=config,
        summary_df=summary_df,
        target_state=target_state,
        best_record=best_record,
        average_record=average_record,
    )

    plot_representative_trajectories(
        config=config,
        best_traj=best_traj,
        average_traj=average_traj,
        target_state=target_state,
        best_record=best_record,
        average_record=average_record,
    )

    print()
    print("Sweep complete")
    print("--------------")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Metadata JSON: {metadata_json}")

    n_ok = int((summary_df["status"] == "OK").sum())
    n_fail = int((summary_df["status"] != "OK").sum())

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
        print(
            "pos err [km]:  "
            f"{best_record['final_position_error_norm_km']:.6e}"
        )
        print(
            "vel err [m/s]: "
            f"{best_record['final_velocity_error_norm_mps']:.6e}"
        )

    if average_record is not None:
        print()
        print("Average/median proxy-cost case")
        print("------------------------------")
        print(f"sample_id:     {int(average_record['sample_id'])}")
        print(f"proxy_cost:    {average_record['proxy_cost']:.6e}")


if __name__ == "__main__":
    main()