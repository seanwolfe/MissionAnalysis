"""
optimize_mcc_from_ballistic_sweep.py

Purpose
-------
Read a ballistic cislunar release sweep, select a subset of cases, and optimize
fixed-time impulsive maneuvers:

    1. Mid-course correction at a fixed fraction of TOF.
    2. Arrival/insertion maneuver at the LPF target epoch.

This script assumes the initial release state is fixed for each selected
ballistic sample. It does not optimize release geometry, TOF, or MCC timing.

State convention
----------------
All states are Earth-centered J2000/EME:

    [x, y, z, vx, vy, vz]

position [km], velocity [km/s].

Inputs
------
- tli_departure_config.yaml
- ballistic_release_sweep_summary.csv
- LPF_orbit_2.csv
- config.yaml constants file for NBodyPropagator

Outputs
-------
- maneuver_optimized_sweep_summary.csv
- maneuver_optimization_metadata.json
- optimized representative trajectories, optionally
"""

import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import spiceypy as spice
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

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
# Target loading
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
# Earth-Moon basis and release-state reconstruction
# =============================================================================

def get_moon_state_geo_j2000(t_jdtdb):
    et = jdtdb_to_et(t_jdtdb)

    moon_state, _ = spice.spkgeo(
        301,
        et,
        "J2000",
        399,
    )

    moon_state = np.asarray(moon_state, dtype=float)

    return moon_state[:3], moon_state[3:]


def get_earth_moon_basis(t_jdtdb):
    r_moon, v_moon = get_moon_state_geo_j2000(t_jdtdb)

    e1 = unit(r_moon)
    e3 = unit(np.cross(r_moon, v_moon))
    e2 = unit(np.cross(e3, e1))

    return e1, e2, e3, r_moon, v_moon


def build_tli_like_state_from_row(row, basis):
    """
    Reconstruct the initial release state from one ballistic-sweep summary row.

    Required row columns:
        r0_mag_km
        C3_km2_s2
        phi_r_deg
        phi_v_deg
        out_of_plane_r_deg
        out_of_plane_v_deg
    """

    e1, e2, e3, r_moon, v_moon = basis

    r0_mag = float(row["r0_mag_km"])
    C3 = float(row["C3_km2_s2"])

    phi_r = np.deg2rad(float(row["phi_r_deg"]))
    phi_v = np.deg2rad(float(row["phi_v_deg"]))
    out_r = np.deg2rad(float(row["out_of_plane_r_deg"]))
    out_v = np.deg2rad(float(row["out_of_plane_v_deg"]))

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
# Propagator setup
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


# =============================================================================
# Case selection
# =============================================================================

def select_cases_for_optimization(summary_df, config):
    opt_cfg = config["maneuver_optimization"]

    df = summary_df.copy()

    if opt_cfg.get("require_ballistic_status_ok", True):
        df = df[df["status"] == "OK"].copy()

    ranking_col = opt_cfg.get("ranking_column", "proxy_cost")
    ascending = bool(opt_cfg.get("ascending", True))

    if ranking_col not in df.columns:
        raise KeyError(f"Ranking column '{ranking_col}' not found in summary CSV.")

    df = df.dropna(subset=[ranking_col])
    df = df.sort_values(ranking_col, ascending=ascending).reset_index(drop=True)

    run_mode = opt_cfg.get("run_mode", "top_n")

    if run_mode == "all":
        selected = df

    elif run_mode == "top_n":
        n = int(opt_cfg.get("top_n", 25))
        selected = df.head(n)

    elif run_mode == "selected_sample_ids":
        ids = set(int(x) for x in opt_cfg.get("selected_sample_ids", []))
        selected = df[df["sample_id"].astype(int).isin(ids)]

    elif run_mode == "random_n":
        n = int(opt_cfg.get("random_n", 50))
        seed = int(opt_cfg.get("random_seed", 1))
        n = min(n, len(df))
        selected = df.sample(n=n, random_state=seed)

    elif run_mode == "percentile_cases":
        percentiles = opt_cfg.get("percentiles", [0, 25, 50, 75, 100])

        selected_rows = []
        values = df[ranking_col].to_numpy(dtype=float)

        for p in percentiles:
            target_value = np.percentile(values, float(p))
            idx = int(np.argmin(np.abs(values - target_value)))
            selected_rows.append(df.iloc[idx])

        selected = pd.DataFrame(selected_rows).drop_duplicates(subset=["sample_id"])

    elif run_mode == "cost_threshold":
        max_cost = float(opt_cfg["max_proxy_cost"])
        selected = df[df[ranking_col] <= max_cost]

    else:
        raise ValueError(f"Unknown maneuver_optimization.run_mode: {run_mode}")

    selected = selected.sort_values(ranking_col, ascending=ascending).reset_index(drop=True)

    print("Maneuver optimization case selection")
    print("------------------------------------")
    print(f"Run mode:         {run_mode}")
    print(f"Available cases:  {len(df)}")
    print(f"Selected cases:   {len(selected)}")
    print()

    return selected


# =============================================================================
# Maneuver model
# =============================================================================

def unpack_maneuver_vector(x):
    """
    Decision vector:

        x = [
            dvm_x, dvm_y, dvm_z,
            dva_x, dva_y, dva_z,
        ]

    Units: km/s.
    """

    x = np.asarray(x, dtype=float)

    dvm = x[0:3]
    dva = x[3:6]

    return dvm, dva


def propagate_with_mcc_and_arrival(
    propagator,
    state0,
    t_depart_jdtdb,
    t_target_jdtdb,
    mcc_fraction,
    dvm,
    dva,
):
    """
    Propagate with one fixed-time MCC and one arrival insertion impulse.

    Sequence:
        state0
        -> coast to MCC
        -> apply dvm
        -> coast to target epoch
        -> apply dva
    """

    t_mcc_jdtdb = float(t_depart_jdtdb) + float(mcc_fraction) * (
        float(t_target_jdtdb) - float(t_depart_jdtdb)
    )

    state_mcc_minus = propagator.propagate(
        x0_km=state0,
        t0_jdtdb=t_depart_jdtdb,
        t1_jdtdb=t_mcc_jdtdb,
    )

    state_mcc_plus = state_mcc_minus.copy()
    state_mcc_plus[3:] += dvm

    state_arrival_minus = propagator.propagate(
        x0_km=state_mcc_plus,
        t0_jdtdb=t_mcc_jdtdb,
        t1_jdtdb=t_target_jdtdb,
    )

    state_arrival_plus = state_arrival_minus.copy()
    state_arrival_plus[3:] += dva

    event_states = {
        "t_mcc_jdtdb": float(t_mcc_jdtdb),
        "state_mcc_minus": state_mcc_minus,
        "state_mcc_plus": state_mcc_plus,
        "state_arrival_minus": state_arrival_minus,
        "state_arrival_plus": state_arrival_plus,
    }

    return state_arrival_plus, event_states


def maneuver_residual(
    x,
    propagator,
    state0,
    target_state,
    t_depart_jdtdb,
    t_target_jdtdb,
    config,
):
    dvm, dva = unpack_maneuver_vector(x)

    mcc_fraction = float(config["maneuvers"]["mcc_fraction"])

    state_final_plus, _ = propagate_with_mcc_and_arrival(
        propagator=propagator,
        state0=state0,
        t_depart_jdtdb=t_depart_jdtdb,
        t_target_jdtdb=t_target_jdtdb,
        mcc_fraction=mcc_fraction,
        dvm=dvm,
        dva=dva,
    )

    dr = state_final_plus[:3] - target_state[:3]
    dv = state_final_plus[3:] - target_state[3:]

    opt_cfg = config["optimization"]

    r_scale = float(opt_cfg["r_scale_km"])
    v_scale = float(opt_cfg["v_scale_kms"])
    dv_scale = float(opt_cfg["dv_scale_kms"])

    residual = [
        dr[0] / r_scale,
        dr[1] / r_scale,
        dr[2] / r_scale,
        dv[0] / v_scale,
        dv[1] / v_scale,
        dv[2] / v_scale,
    ]

    if bool(opt_cfg.get("include_dv_penalty", True)):
        dv_total = np.linalg.norm(dvm) + np.linalg.norm(dva)
        residual.append(dv_total / dv_scale)

    return np.asarray(residual, dtype=float)


def optimize_maneuvers_for_case(
    row,
    basis,
    propagator,
    target_state,
    t_depart_jdtdb,
    t_target_jdtdb,
    config,
):
    """
    Optimize MCC and arrival insertion for one selected ballistic case.
    """

    state0 = build_tli_like_state_from_row(row, basis)

    initial_guess = float(config["maneuvers"].get("initial_guess_kms", 0.0))
    x0 = np.full(6, initial_guess, dtype=float)

    bound = float(config["maneuvers"]["dv_component_bound_kms"])
    lower = -bound * np.ones(6)
    upper = +bound * np.ones(6)

    opt_cfg = config["optimization"]

    result = least_squares(
        maneuver_residual,
        x0,
        bounds=(lower, upper),
        args=(
            propagator,
            state0,
            target_state,
            t_depart_jdtdb,
            t_target_jdtdb,
            config,
        ),
        max_nfev=int(opt_cfg.get("max_nfev", 150)),
        xtol=float(opt_cfg.get("xtol", 1e-10)),
        ftol=float(opt_cfg.get("ftol", 1e-10)),
        gtol=float(opt_cfg.get("gtol", 1e-10)),
        verbose=0,
    )

    dvm, dva = unpack_maneuver_vector(result.x)

    final_state_plus, event_states = propagate_with_mcc_and_arrival(
        propagator=propagator,
        state0=state0,
        t_depart_jdtdb=t_depart_jdtdb,
        t_target_jdtdb=t_target_jdtdb,
        mcc_fraction=float(config["maneuvers"]["mcc_fraction"]),
        dvm=dvm,
        dva=dva,
    )

    dr_final = final_state_plus[:3] - target_state[:3]
    dv_final = final_state_plus[3:] - target_state[3:]

    dvm_mag = np.linalg.norm(dvm)
    dva_mag = np.linalg.norm(dva)

    output = {
        "sample_id": int(row["sample_id"]),
        "optimization_success": bool(result.success),
        "optimization_status": int(result.status),
        "optimization_message": str(result.message),
        "nfev": int(result.nfev),
        "least_squares_cost": float(result.cost),

        "ballistic_proxy_cost": float(row.get("proxy_cost", np.nan)),
        "ballistic_position_error_norm_km": float(row.get("final_position_error_norm_km", np.nan)),
        "ballistic_velocity_error_norm_mps": float(row.get("final_velocity_error_norm_mps", np.nan)),

        "r0_mag_km": float(row["r0_mag_km"]),
        "C3_km2_s2": float(row["C3_km2_s2"]),
        "phi_r_deg": float(row["phi_r_deg"]),
        "phi_v_deg": float(row["phi_v_deg"]),
        "out_of_plane_r_deg": float(row["out_of_plane_r_deg"]),
        "out_of_plane_v_deg": float(row["out_of_plane_v_deg"]),

        "initial_x_geo_j2000_km": float(state0[0]),
        "initial_y_geo_j2000_km": float(state0[1]),
        "initial_z_geo_j2000_km": float(state0[2]),
        "initial_vx_geo_j2000_kms": float(state0[3]),
        "initial_vy_geo_j2000_kms": float(state0[4]),
        "initial_vz_geo_j2000_kms": float(state0[5]),
        "initial_C3_km2_s2": float(compute_C3_geo(state0)),

        "t_mcc_jdtdb": float(event_states["t_mcc_jdtdb"]),
        "t_mcc_utc": jdtdb_to_utc(event_states["t_mcc_jdtdb"]),

        "dvm_x_kms": float(dvm[0]),
        "dvm_y_kms": float(dvm[1]),
        "dvm_z_kms": float(dvm[2]),
        "dva_x_kms": float(dva[0]),
        "dva_y_kms": float(dva[1]),
        "dva_z_kms": float(dva[2]),

        "dvm_mag_mps": float(1000.0 * dvm_mag),
        "dva_mag_mps": float(1000.0 * dva_mag),
        "dv_total_mps": float(1000.0 * (dvm_mag + dva_mag)),

        "final_x_geo_j2000_km": float(final_state_plus[0]),
        "final_y_geo_j2000_km": float(final_state_plus[1]),
        "final_z_geo_j2000_km": float(final_state_plus[2]),
        "final_vx_geo_j2000_kms": float(final_state_plus[3]),
        "final_vy_geo_j2000_kms": float(final_state_plus[4]),
        "final_vz_geo_j2000_kms": float(final_state_plus[5]),

        "final_position_error_norm_km": float(np.linalg.norm(dr_final)),
        "final_velocity_error_norm_mps": float(1000.0 * np.linalg.norm(dv_final)),
        "final_position_error_x_km": float(dr_final[0]),
        "final_position_error_y_km": float(dr_final[1]),
        "final_position_error_z_km": float(dr_final[2]),
        "final_velocity_error_x_mps": float(1000.0 * dv_final[0]),
        "final_velocity_error_y_mps": float(1000.0 * dv_final[1]),
        "final_velocity_error_z_mps": float(1000.0 * dv_final[2]),
    }

    return output, result, state0


# =============================================================================
# Optimized trajectory reconstruction
# =============================================================================

def propagate_optimized_trajectory_for_output(
    propagator,
    state0,
    dvm,
    dva,
    t_depart_jdtdb,
    t_target_jdtdb,
    config,
):
    """
    Repropagate optimized trajectory and return a continuous trajectory dataframe.

    The CSV includes duplicated event labels but not duplicate epochs except the
    impulse states are represented by event rows in a separate table.
    """

    step_days = float(config["propagation"]["step_days"])
    mcc_fraction = float(config["maneuvers"]["mcc_fraction"])

    t_mcc_jdtdb = float(t_depart_jdtdb) + mcc_fraction * (
        float(t_target_jdtdb) - float(t_depart_jdtdb)
    )

    # Segment 1
    t_grid_1 = build_time_grid(t_depart_jdtdb, t_mcc_jdtdb, step_days)

    X1 = propagator.propagate(
        x0_km=state0,
        t0_jdtdb=t_depart_jdtdb,
        t1_jdtdb=t_grid_1,
    )

    # Apply MCC
    state_mcc_plus = X1[-1].copy()
    state_mcc_plus[3:] += dvm

    # Segment 2
    t_grid_2 = build_time_grid(t_mcc_jdtdb, t_target_jdtdb, step_days)

    X2 = propagator.propagate(
        x0_km=state_mcc_plus,
        t0_jdtdb=t_mcc_jdtdb,
        t1_jdtdb=t_grid_2,
    )

    # Avoid duplicate t_mcc row in continuous trajectory
    t_all = np.hstack([t_grid_1, t_grid_2[1:]])
    X_all = np.vstack([X1, X2[1:]])

    r = X_all[:, :3]
    v = X_all[:, 3:]

    r_norm = np.linalg.norm(r, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    C3 = v_norm**2 - 2.0 * MU_EARTH_KM3_S2 / r_norm

    df = pd.DataFrame({
        "jdtdb": t_all,
        "utc": [jdtdb_to_utc(t) for t in t_all],
        "t_days_since_departure": t_all - float(t_depart_jdtdb),

        "x_geo_j2000_km": X_all[:, 0],
        "y_geo_j2000_km": X_all[:, 1],
        "z_geo_j2000_km": X_all[:, 2],
        "vx_geo_j2000_kms": X_all[:, 3],
        "vy_geo_j2000_kms": X_all[:, 4],
        "vz_geo_j2000_kms": X_all[:, 5],

        "r_geo_km": r_norm,
        "v_geo_kms": v_norm,
        "C3_geo_km2_s2": C3,
    })

    event_rows = []

    event_rows.append({
        "event": "departure",
        "jdtdb": float(t_depart_jdtdb),
        "utc": jdtdb_to_utc(t_depart_jdtdb),
        "dvx_kms": 0.0,
        "dvy_kms": 0.0,
        "dvz_kms": 0.0,
        "dv_mag_mps": 0.0,
    })

    event_rows.append({
        "event": "mcc",
        "jdtdb": float(t_mcc_jdtdb),
        "utc": jdtdb_to_utc(t_mcc_jdtdb),
        "dvx_kms": float(dvm[0]),
        "dvy_kms": float(dvm[1]),
        "dvz_kms": float(dvm[2]),
        "dv_mag_mps": float(1000.0 * np.linalg.norm(dvm)),
    })

    event_rows.append({
        "event": "arrival_insertion",
        "jdtdb": float(t_target_jdtdb),
        "utc": jdtdb_to_utc(t_target_jdtdb),
        "dvx_kms": float(dva[0]),
        "dvy_kms": float(dva[1]),
        "dvz_kms": float(dva[2]),
        "dv_mag_mps": float(1000.0 * np.linalg.norm(dva)),
    })

    event_df = pd.DataFrame(event_rows)

    return df, event_df


# =============================================================================
# Save outputs
# =============================================================================

def save_optimization_outputs(
    config,
    opt_summary_df,
    metadata,
):
    out_dir = Path(config["maneuver_optimization"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    output_cfg = config.get("output", {})

    summary_path = out_dir / "maneuver_optimized_sweep_summary.csv"
    metadata_path = out_dir / "maneuver_optimization_metadata.json"

    if bool(output_cfg.get("save_optimization_summary_csv", True)):
        opt_summary_df.to_csv(summary_path, index=False)

    if bool(output_cfg.get("save_optimization_metadata_json", True)):
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return summary_path, metadata_path


def save_representative_optimized_trajectories(
    config,
    opt_summary_df,
    propagator,
    basis,
    t_depart_jdtdb,
    t_target_jdtdb,
):
    """
    Save representative optimized trajectories:
        - minimum total ΔV
        - median total ΔV
        - maximum total ΔV

    Only successful optimizer rows are considered.
    """

    if not bool(config.get("output", {}).get("save_representative_trajectories", True)):
        return {}

    out_dir = Path(config["maneuver_optimization"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = opt_summary_df[opt_summary_df["optimization_success"] == True].copy()
    ok = ok.dropna(subset=["dv_total_mps"])

    if ok.empty:
        return {}

    ok = ok.sort_values("dv_total_mps").reset_index(drop=True)

    representatives = {}

    best_row = ok.iloc[0]
    median_dv = ok["dv_total_mps"].median()
    median_idx = int(np.argmin(np.abs(ok["dv_total_mps"].to_numpy() - median_dv)))
    median_row = ok.iloc[median_idx]
    worst_row = ok.iloc[-1]

    reps = {
        "best_dv": best_row,
        "median_dv": median_row,
        "worst_dv": worst_row,
    }

    for name, row in reps.items():
        state0 = build_tli_like_state_from_row(row, basis)

        dvm = np.array([
            row["dvm_x_kms"],
            row["dvm_y_kms"],
            row["dvm_z_kms"],
        ], dtype=float)

        dva = np.array([
            row["dva_x_kms"],
            row["dva_y_kms"],
            row["dva_z_kms"],
        ], dtype=float)

        traj_df, event_df = propagate_optimized_trajectory_for_output(
            propagator=propagator,
            state0=state0,
            dvm=dvm,
            dva=dva,
            t_depart_jdtdb=t_depart_jdtdb,
            t_target_jdtdb=t_target_jdtdb,
            config=config,
        )

        traj_path = out_dir / f"optimized_{name}_trajectory.csv"
        event_path = out_dir / f"optimized_{name}_maneuvers.csv"

        traj_df.to_csv(traj_path, index=False)
        event_df.to_csv(event_path, index=False)

        representatives[name] = {
            "sample_id": int(row["sample_id"]),
            "dv_total_mps": float(row["dv_total_mps"]),
            "trajectory_csv": str(traj_path),
            "maneuver_csv": str(event_path),
        }

    return representatives


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


def plot_representative_optimized_trajectories(config, target_state, representatives):
    viz = config.get("visualization", {}).get("optimized_representative_trajectories", {})

    if not bool(viz.get("enabled", False)):
        return

    out_dir = Path(config["maneuver_optimization"]["output_dir"])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    all_points = [
        np.zeros((1, 3)),
        np.asarray(target_state[:3]).reshape(1, 3),
    ]

    plot_options = {
        "best_dv": bool(viz.get("include_best_dv", True)),
        "median_dv": bool(viz.get("include_median_dv", True)),
        "worst_dv": bool(viz.get("include_worst_dv", False)),
    }

    labels = {
        "best_dv": "Best ΔV",
        "median_dv": "Median ΔV",
        "worst_dv": "Worst ΔV",
    }

    for key, enabled in plot_options.items():
        if not enabled:
            continue
        if key not in representatives:
            continue

        traj_path = representatives[key]["trajectory_csv"]
        sample_id = representatives[key]["sample_id"]
        dv_total = representatives[key]["dv_total_mps"]

        df = pd.read_csv(traj_path)
        r = df[["x_geo_j2000_km", "y_geo_j2000_km", "z_geo_j2000_km"]].to_numpy()

        all_points.append(r)

        ax.plot(
            r[:, 0],
            r[:, 1],
            r[:, 2],
            label=f"{labels[key]} case, id={sample_id}, ΔV={dv_total:.1f} m/s",
        )

        ax.scatter([r[0, 0]], [r[0, 1]], [r[0, 2]], marker="o", s=50)
        ax.scatter([r[-1, 0]], [r[-1, 1]], [r[-1, 2]], marker="x", s=70)

    ax.scatter([0.0], [0.0], [0.0], s=80, label="Earth")

    ax.scatter(
        [target_state[0]],
        [target_state[1]],
        [target_state[2]],
        marker="*",
        s=100,
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

    ax.set_title("Optimized MCC + arrival insertion representative trajectories")
    ax.legend(fontsize=8)

    set_axes_equal_3d(ax)
    plt.tight_layout()

    if bool(viz.get("save_plot", True)):
        filename = viz.get("filename", "optimized_representative_trajectories.png")
        fig.savefig(out_dir / filename, dpi=300)

    if bool(viz.get("show_plot", False)):
        plt.show()

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    config_path = "transfer_tli_departure_config_monte.yaml"
    config = load_config(config_path)

    if not bool(config.get("maneuver_optimization", {}).get("enabled", True)):
        print("maneuver_optimization.enabled is false. Exiting.")
        return

    load_spice_kernels(config)

    # -------------------------------------------------------------------------
    # Target and timing
    # -------------------------------------------------------------------------
    target_time_utc, target_state, target_row = load_lpf_target_state(config)

    tof_days = float(config["epoch"]["tof_days"])
    t_target_jdtdb = utc_to_jdtdb(target_time_utc)
    t_depart_jdtdb = t_target_jdtdb - tof_days

    print("MCC/insertion optimization")
    print("--------------------------")
    print(f"Target index:     {config['target']['target_index']}")
    print(f"Target UTC:       {jdtdb_to_utc(t_target_jdtdb)}")
    print(f"Departure UTC:    {jdtdb_to_utc(t_depart_jdtdb)}")
    print(f"TOF days:         {tof_days:.6f}")
    print(f"MCC fraction:     {float(config['maneuvers']['mcc_fraction']):.3f}")
    print()

    # -------------------------------------------------------------------------
    # Load ballistic summary and select cases
    # -------------------------------------------------------------------------
    input_summary_csv = config["maneuver_optimization"]["input_summary_csv"]
    ballistic_df = pd.read_csv(input_summary_csv)

    selected_df = select_cases_for_optimization(ballistic_df, config)

    if selected_df.empty:
        raise RuntimeError("No cases selected for maneuver optimization.")

    # -------------------------------------------------------------------------
    # Setup propagator and basis
    # -------------------------------------------------------------------------
    basis = get_earth_moon_basis(t_depart_jdtdb)
    propagator = build_nbody_propagator(config)

    # -------------------------------------------------------------------------
    # Optimize selected cases
    # -------------------------------------------------------------------------
    records = []

    for i, row in selected_df.iterrows():
        sample_id = int(row["sample_id"])

        print(f"Optimizing case {i + 1}/{len(selected_df)}: sample_id={sample_id}")

        try:
            record, result, state0 = optimize_maneuvers_for_case(
                row=row,
                basis=basis,
                propagator=propagator,
                target_state=target_state,
                t_depart_jdtdb=t_depart_jdtdb,
                t_target_jdtdb=t_target_jdtdb,
                config=config,
            )

        except Exception as e:
            record = {
                "sample_id": sample_id,
                "optimization_success": False,
                "optimization_status": -999,
                "optimization_message": str(e),
                "nfev": 0,
                "least_squares_cost": np.nan,

                "ballistic_proxy_cost": float(row.get("proxy_cost", np.nan)),
                "ballistic_position_error_norm_km": float(row.get("final_position_error_norm_km", np.nan)),
                "ballistic_velocity_error_norm_mps": float(row.get("final_velocity_error_norm_mps", np.nan)),

                "r0_mag_km": float(row["r0_mag_km"]),
                "C3_km2_s2": float(row["C3_km2_s2"]),
                "phi_r_deg": float(row["phi_r_deg"]),
                "phi_v_deg": float(row["phi_v_deg"]),
                "out_of_plane_r_deg": float(row["out_of_plane_r_deg"]),
                "out_of_plane_v_deg": float(row["out_of_plane_v_deg"]),

                "dvm_x_kms": np.nan,
                "dvm_y_kms": np.nan,
                "dvm_z_kms": np.nan,
                "dva_x_kms": np.nan,
                "dva_y_kms": np.nan,
                "dva_z_kms": np.nan,

                "dvm_mag_mps": np.nan,
                "dva_mag_mps": np.nan,
                "dv_total_mps": np.nan,

                "final_position_error_norm_km": np.nan,
                "final_velocity_error_norm_mps": np.nan,
            }

        records.append(record)

    opt_summary_df = pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    n_success = int((opt_summary_df["optimization_success"] == True).sum())
    n_fail = int((opt_summary_df["optimization_success"] != True).sum())

    metadata = {
        "input_summary_csv": input_summary_csv,
        "n_selected_cases": int(len(selected_df)),
        "n_success": n_success,
        "n_fail": n_fail,

        "target_index": int(config["target"]["target_index"]),
        "target_time_jdtdb": float(t_target_jdtdb),
        "target_time_utc": jdtdb_to_utc(t_target_jdtdb),
        "departure_time_jdtdb": float(t_depart_jdtdb),
        "departure_time_utc": jdtdb_to_utc(t_depart_jdtdb),
        "tof_days": float(tof_days),
        "mcc_fraction": float(config["maneuvers"]["mcc_fraction"]),

        "target_state_geo_eme_km_kms": target_state.tolist(),
    }

    summary_path, metadata_path = save_optimization_outputs(
        config=config,
        opt_summary_df=opt_summary_df,
        metadata=metadata,
    )

    representatives = save_representative_optimized_trajectories(
        config=config,
        opt_summary_df=opt_summary_df,
        propagator=propagator,
        basis=basis,
        t_depart_jdtdb=t_depart_jdtdb,
        t_target_jdtdb=t_target_jdtdb,
    )

    metadata["representative_trajectories"] = representatives

    # Re-save metadata after adding representatives
    out_dir = Path(config["maneuver_optimization"]["output_dir"])
    with open(out_dir / "maneuver_optimization_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    plot_representative_optimized_trajectories(
        config=config,
        target_state=target_state,
        representatives=representatives,
    )

    print()
    print("Optimization complete")
    print("---------------------")
    print(f"Summary CSV:   {summary_path}")
    print(f"Metadata JSON: {metadata_path}")
    print(f"Successful:    {n_success}")
    print(f"Failed:        {n_fail}")

    ok = opt_summary_df[opt_summary_df["optimization_success"] == True].copy()
    ok = ok.dropna(subset=["dv_total_mps"])

    if not ok.empty:
        print()
        print("ΔV summary for successful cases")
        print("-------------------------------")
        print(f"min total ΔV [m/s]:    {ok['dv_total_mps'].min():.3f}")
        print(f"median total ΔV [m/s]: {ok['dv_total_mps'].median():.3f}")
        print(f"mean total ΔV [m/s]:   {ok['dv_total_mps'].mean():.3f}")
        print(f"max total ΔV [m/s]:    {ok['dv_total_mps'].max():.3f}")


if __name__ == "__main__":
    main()