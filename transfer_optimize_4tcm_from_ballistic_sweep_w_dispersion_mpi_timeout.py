"""
optimize_four_burn_with_dispersion.py

Modes
-----
1. input_mode = "ballistic_summary"
   Reconstructs release states from a ballistic sweep summary and optimizes
   the four-burn sequence.

2. input_mode = "dispersion"
   Loads a nominal optimized trajectory from a previous four-burn summary,
   generates Gaussian perturbations around the nominal initial state, and
   optimizes the same four-burn sequence for each dispersed state.

Four-burn sequence
------------------
TCM-1 : injection clean-up / first correction
TCM-2 : early transfer correction
TCM-3 : mid-course correction
ACQ   : quasi-halo acquisition + trim

State convention
----------------
Earth-centered J2000/EME:
    [x, y, z, vx, vy, vz]

position [km], velocity [km/s].
"""

import argparse
import json
import time
from pathlib import Path

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

import yaml
import numpy as np
import pandas as pd
import spiceypy as spice
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from n_body_integrator import NBodyPropagator


MU_EARTH_KM3_S2 = 398600.4418


class OptimizationTimeoutError(RuntimeError):
    """Raised when one optimization case exceeds its configured wallclock limit."""
    pass


# =============================================================================
# Utilities
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
    ts = pd.to_datetime(str(utc_string))
    utc_clean = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")
    et = spice.str2et(utc_clean)
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
    moon_state, _ = spice.spkgeo(301, et, "J2000", 399)
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
    Reconstruct initial release state from a ballistic-sweep-style row.
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


def get_initial_state_from_row(row, basis=None):
    """
    Get initial state either from explicit initial-state columns or from
    TLI release parameters.
    """

    explicit_cols = [
        "initial_x_geo_j2000_km",
        "initial_y_geo_j2000_km",
        "initial_z_geo_j2000_km",
        "initial_vx_geo_j2000_kms",
        "initial_vy_geo_j2000_kms",
        "initial_vz_geo_j2000_kms",
    ]

    if all(c in row.index for c in explicit_cols):
        return row[explicit_cols].to_numpy(dtype=float)

    if basis is None:
        raise ValueError(
            "Row does not contain explicit initial state columns and basis was not provided."
        )

    return build_tli_like_state_from_row(row, basis)


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
# Maneuver sequence
# =============================================================================

def get_enabled_maneuvers(config):
    sequence = config["maneuvers"]["sequence"]
    enabled = [m for m in sequence if bool(m.get("enabled", True))]
    if len(enabled) == 0:
        raise ValueError("No maneuvers are enabled.")

    enabled = sorted(enabled, key=lambda m: float(m["fraction"]))
    fractions = [float(m["fraction"]) for m in enabled]

    if any(f < 0.0 for f in fractions):
        raise ValueError("Maneuver fractions must be non-negative.")

    if fractions[-1] > 1.0:
        raise ValueError(
            "This script assumes all optimized maneuvers occur within the transfer. "
            "Use ACQ at fraction 1.0 for final acquisition."
        )

    return enabled


def maneuver_times_from_sequence(maneuvers, t_depart_jdtdb, t_target_jdtdb):
    tof = float(t_target_jdtdb) - float(t_depart_jdtdb)
    return np.asarray(
        [float(t_depart_jdtdb) + float(m["fraction"]) * tof for m in maneuvers],
        dtype=float,
    )


def unpack_maneuver_vector(x, n_maneuvers):
    x = np.asarray(x, dtype=float)
    if len(x) != 3 * n_maneuvers:
        raise ValueError(f"Expected {3*n_maneuvers} variables, got {len(x)}.")
    return [x[3*k:3*k+3] for k in range(n_maneuvers)]


def build_initial_guess_and_bounds(maneuvers):
    x0 = []
    lower = []
    upper = []

    for m in maneuvers:
        guess = float(m.get("initial_guess_kms", 0.0))
        bound = float(m["dv_component_bound_kms"])

        x0.extend([guess, guess, guess])
        lower.extend([-bound, -bound, -bound])
        upper.extend([+bound, +bound, +bound])

    return np.asarray(x0), np.asarray(lower), np.asarray(upper)


# =============================================================================
# Ballistic-summary mode case selection
# =============================================================================

def select_cases_from_ballistic_summary(summary_df, config):
    opt_cfg = config["four_burn_optimization"]
    df = summary_df.copy()

    if opt_cfg.get("require_ballistic_status_ok", True):
        df = df[df["status"] == "OK"].copy()

    ranking_col = opt_cfg.get("ranking_column", "proxy_cost")
    ascending = bool(opt_cfg.get("ascending", True))

    if ranking_col not in df.columns:
        raise KeyError(f"Ranking column '{ranking_col}' not found.")

    df = df.dropna(subset=[ranking_col])
    df = df.sort_values(ranking_col, ascending=ascending).reset_index(drop=True)

    run_mode = opt_cfg.get("run_mode", "top_n")

    if run_mode == "all":
        selected = df

    elif run_mode == "top_n":
        selected = df.head(int(opt_cfg.get("top_n", 25)))

    elif run_mode == "selected_sample_ids":
        ids = set(int(x) for x in opt_cfg.get("selected_sample_ids", []))
        selected = df[df["sample_id"].astype(int).isin(ids)]

    elif run_mode == "random_n":
        n = min(int(opt_cfg.get("random_n", 50)), len(df))
        seed = int(opt_cfg.get("random_seed", 1))
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
        raise ValueError(f"Unknown run_mode: {run_mode}")

    selected = selected.sort_values(ranking_col, ascending=ascending).reset_index(drop=True)

    selected["case_type"] = "ballistic_grid"
    selected["case_id"] = selected["sample_id"].astype(str)

    print("Ballistic-summary case selection")
    print("--------------------------------")
    print(f"Run mode:         {run_mode}")
    print(f"Available cases:  {len(df)}")
    print(f"Selected cases:   {len(selected)}")
    print()

    return selected


# =============================================================================
# Dispersion mode
# =============================================================================

def select_nominal_optimized_case(config):
    nom_cfg = config["nominal_case"]
    path = config["four_burn_optimization"]["nominal_optimized_summary_csv"]

    df = pd.read_csv(path)

    if bool(nom_cfg.get("require_target_success", True)) and "target_success" in df.columns:
        df = df[df["target_success"] == True].copy()

    if "optimizer_success" in df.columns:
        df = df[df["optimizer_success"] == True].copy()

    if df.empty:
        raise RuntimeError("No valid nominal cases found in optimized summary.")

    df = df.dropna(subset=["dv_total_mps"]).copy()
    df = df.sort_values("dv_total_mps").reset_index(drop=True)

    selection = nom_cfg.get("selection", "best_dv")

    if selection == "best_dv":
        row = df.iloc[0]

    elif selection == "median_dv":
        median_dv = df["dv_total_mps"].median()
        idx = int(np.argmin(np.abs(df["dv_total_mps"].to_numpy() - median_dv)))
        row = df.iloc[idx]

    elif selection == "sample_id":
        sid = int(nom_cfg["sample_id"])
        sub = df[df["sample_id"].astype(int) == sid]
        if sub.empty:
            raise RuntimeError(f"sample_id={sid} not found among valid nominal cases.")
        row = sub.iloc[0]

    else:
        raise ValueError(f"Unknown nominal_case.selection: {selection}")

    print("Nominal optimized case")
    print("----------------------")
    print(f"Selection:       {selection}")
    print(f"sample_id:       {row['sample_id']}")
    print(f"nominal ΔV [m/s]: {row['dv_total_mps']:.3f}")
    print()

    return row


def rtn_basis_from_state(state):
    r = np.asarray(state[:3], dtype=float)
    v = np.asarray(state[3:], dtype=float)

    e_r = unit(r)
    h = np.cross(r, v)
    e_n = unit(h)
    e_t = unit(np.cross(e_n, e_r))

    # Columns map RTN components to J2000.
    C = np.column_stack([e_r, e_t, e_n])
    return C


def draw_clipped_normal(rng, sigma, clip_sigma=None):
    sigma = np.asarray(sigma, dtype=float)

    if clip_sigma is None:
        return rng.normal(0.0, sigma)

    out = rng.normal(0.0, sigma)
    lim = float(clip_sigma) * sigma

    # Component-wise clipping.
    return np.clip(out, -lim, +lim)


def generate_dispersion_cases(config, nominal_row):
    disp_cfg = config["dispersion"]

    n_samples = int(disp_cfg["n_samples"])
    seed = int(disp_cfg.get("random_seed", 1))
    rng = np.random.default_rng(seed)

    frame = disp_cfg.get("frame", "rtn").lower()
    clip_sigma = disp_cfg.get("clip_sigma", None)

    if clip_sigma is not None:
        clip_sigma = float(clip_sigma)

    nominal_state = get_initial_state_from_row(nominal_row)

    pos_sigma = np.array([
        float(disp_cfg["position_sigma_km"]["x"]),
        float(disp_cfg["position_sigma_km"]["y"]),
        float(disp_cfg["position_sigma_km"]["z"]),
    ])

    vel_sigma_kms = np.array([
        float(disp_cfg["velocity_sigma_mps"]["x"]),
        float(disp_cfg["velocity_sigma_mps"]["y"]),
        float(disp_cfg["velocity_sigma_mps"]["z"]),
    ]) / 1000.0

    if frame == "rtn":
        C = rtn_basis_from_state(nominal_state)
    elif frame == "j2000":
        C = np.eye(3)
    else:
        raise ValueError("dispersion.frame must be 'rtn' or 'j2000'.")

    rows = []

    for k in range(n_samples):
        dpos_local = draw_clipped_normal(rng, pos_sigma, clip_sigma)
        dvel_local = draw_clipped_normal(rng, vel_sigma_kms, clip_sigma)

        dpos_j2000 = C @ dpos_local
        dvel_j2000 = C @ dvel_local

        state = nominal_state.copy()
        state[:3] += dpos_j2000
        state[3:] += dvel_j2000

        row = nominal_row.copy()

        row["case_type"] = "dispersion"
        row["case_id"] = f"disp_{k:05d}"
        row["sample_id"] = k
        row["nominal_source_sample_id"] = nominal_row["sample_id"]

        row["initial_x_geo_j2000_km"] = state[0]
        row["initial_y_geo_j2000_km"] = state[1]
        row["initial_z_geo_j2000_km"] = state[2]
        row["initial_vx_geo_j2000_kms"] = state[3]
        row["initial_vy_geo_j2000_kms"] = state[4]
        row["initial_vz_geo_j2000_kms"] = state[5]

        row["dispersion_frame"] = frame

        row["dpos_local_x_km"] = dpos_local[0]
        row["dpos_local_y_km"] = dpos_local[1]
        row["dpos_local_z_km"] = dpos_local[2]

        row["dvel_local_x_mps"] = 1000.0 * dvel_local[0]
        row["dvel_local_y_mps"] = 1000.0 * dvel_local[1]
        row["dvel_local_z_mps"] = 1000.0 * dvel_local[2]

        row["dpos_j2000_x_km"] = dpos_j2000[0]
        row["dpos_j2000_y_km"] = dpos_j2000[1]
        row["dpos_j2000_z_km"] = dpos_j2000[2]

        row["dvel_j2000_x_mps"] = 1000.0 * dvel_j2000[0]
        row["dvel_j2000_y_mps"] = 1000.0 * dvel_j2000[1]
        row["dvel_j2000_z_mps"] = 1000.0 * dvel_j2000[2]

        row["dispersion_position_norm_km"] = float(np.linalg.norm(dpos_j2000))
        row["dispersion_velocity_norm_mps"] = float(1000.0 * np.linalg.norm(dvel_j2000))

        rows.append(row)

    df = pd.DataFrame(rows)

    print("Generated dispersion cases")
    print("--------------------------")
    print(f"Samples: {len(df)}")
    print(f"Frame:   {frame}")
    print()

    return df


def build_input_cases(config):
    input_mode = config["four_burn_optimization"].get("input_mode", "ballistic_summary")

    if input_mode == "ballistic_summary":
        path = config["four_burn_optimization"]["input_summary_csv"]
        df = pd.read_csv(path)
        return select_cases_from_ballistic_summary(df, config), None

    if input_mode == "dispersion":
        nominal_row = select_nominal_optimized_case(config)
        return generate_dispersion_cases(config, nominal_row), nominal_row

    raise ValueError(f"Unknown four_burn_optimization.input_mode: {input_mode}")


# =============================================================================
# Propagation and residual
# =============================================================================

def propagate_with_maneuver_sequence(
    propagator,
    state0,
    t_depart_jdtdb,
    t_target_jdtdb,
    maneuver_times,
    dv_list,
):
    state = np.asarray(state0, dtype=float).copy()
    current_t = float(t_depart_jdtdb)
    event_records = []

    for k, (t_burn, dv) in enumerate(zip(maneuver_times, dv_list)):
        t_burn = float(t_burn)
        dv = np.asarray(dv, dtype=float)

        if t_burn > current_t:
            state_minus = propagator.propagate(
                x0_km=state,
                t0_jdtdb=current_t,
                t1_jdtdb=t_burn,
            )
        else:
            state_minus = state.copy()

        state_plus = state_minus.copy()
        state_plus[3:] += dv

        event_records.append({
            "event_index": k,
            "t_jdtdb": t_burn,
            "state_minus": state_minus.copy(),
            "state_plus": state_plus.copy(),
            "dv": dv.copy(),
        })

        state = state_plus
        current_t = t_burn

    if current_t < float(t_target_jdtdb):
        state = propagator.propagate(
            x0_km=state,
            t0_jdtdb=current_t,
            t1_jdtdb=t_target_jdtdb,
        )

    return state, event_records


def maneuver_residual(
    x,
    propagator,
    state0,
    target_state,
    t_depart_jdtdb,
    t_target_jdtdb,
    maneuvers,
    maneuver_times,
    config,
    case_deadline_time=None,
):
    if case_deadline_time is not None and time.time() >= float(case_deadline_time):
        raise OptimizationTimeoutError(
            "Optimization case exceeded runtime.max_optimization_minutes."
        )

    n_maneuvers = len(maneuvers)
    dv_list = unpack_maneuver_vector(x, n_maneuvers)

    final_state, event_records = propagate_with_maneuver_sequence(
        propagator,
        state0,
        t_depart_jdtdb,
        t_target_jdtdb,
        maneuver_times,
        dv_list,
    )

    if case_deadline_time is not None and time.time() >= float(case_deadline_time):
        raise OptimizationTimeoutError(
            "Optimization case exceeded runtime.max_optimization_minutes."
        )

    dr = final_state[:3] - target_state[:3]
    dv = final_state[3:] - target_state[3:]

    opt_cfg = config["optimization"]

    r_scale = float(opt_cfg["r_scale_km"])
    v_scale = float(opt_cfg["v_scale_kms"])
    dv_scale = float(opt_cfg.get("dv_scale_kms", 1.0))
    total_dv_scale = float(opt_cfg.get("total_dv_scale_kms", dv_scale))

    residual = []
    residual.extend((dr / r_scale).tolist())
    residual.extend((dv / v_scale).tolist())

    if bool(opt_cfg.get("include_per_maneuver_dv_penalty", False)):
        for m, dv_k in zip(maneuvers, dv_list):
            weight = float(m.get("dv_weight", 1.0))
            residual.append(weight * np.linalg.norm(dv_k) / dv_scale)

    if bool(opt_cfg.get("include_total_dv_penalty", False)):
        dv_total = sum(np.linalg.norm(dv_k) for dv_k in dv_list)
        residual.append(dv_total / total_dv_scale)

    return np.asarray(residual, dtype=float)


# =============================================================================
# Optimization
# =============================================================================

def optimize_four_burn_for_case(
    row,
    basis,
    propagator,
    target_state,
    t_depart_jdtdb,
    t_target_jdtdb,
    config,
):
    state0 = get_initial_state_from_row(row, basis=basis)

    maneuvers = get_enabled_maneuvers(config)
    maneuver_times = maneuver_times_from_sequence(maneuvers, t_depart_jdtdb, t_target_jdtdb)

    x0, lower, upper = build_initial_guess_and_bounds(maneuvers)
    opt_cfg = config["optimization"]

    case_start_time = time.time()
    max_minutes = config.get("runtime", {}).get("max_optimization_minutes", None)
    if max_minutes is None:
        case_deadline_time = None
    else:
        max_minutes = float(max_minutes)
        case_deadline_time = None if max_minutes <= 0.0 else case_start_time + 60.0 * max_minutes

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
            maneuvers,
            maneuver_times,
            config,
            case_deadline_time,
        ),
        max_nfev=int(opt_cfg.get("max_nfev", 300)),
        xtol=float(opt_cfg.get("xtol", 1e-10)),
        ftol=float(opt_cfg.get("ftol", 1e-10)),
        gtol=float(opt_cfg.get("gtol", 1e-10)),
        verbose=0,
    )

    if case_deadline_time is not None and time.time() >= float(case_deadline_time):
        raise OptimizationTimeoutError(
            "Optimization case exceeded runtime.max_optimization_minutes."
        )

    optimization_runtime_seconds = time.time() - case_start_time

    dv_list = unpack_maneuver_vector(result.x, len(maneuvers))

    final_state, event_records = propagate_with_maneuver_sequence(
        propagator,
        state0,
        t_depart_jdtdb,
        t_target_jdtdb,
        maneuver_times,
        dv_list,
    )

    dr_final = final_state[:3] - target_state[:3]
    dv_final = final_state[3:] - target_state[3:]

    final_pos_err = float(np.linalg.norm(dr_final))
    final_vel_err_mps = float(1000.0 * np.linalg.norm(dv_final))

    pos_tol = float(opt_cfg.get("success_position_tol_km", np.inf))
    vel_tol = float(opt_cfg.get("success_velocity_tol_mps", np.inf))

    target_success = bool(final_pos_err <= pos_tol and final_vel_err_mps <= vel_tol)

    dv_mags_mps = [1000.0 * float(np.linalg.norm(dv_k)) for dv_k in dv_list]
    dv_total_mps = float(np.sum(dv_mags_mps))

    record = {
        "case_id": str(row.get("case_id", row.get("sample_id", ""))),
        "case_type": str(row.get("case_type", "unknown")),
        "sample_id": int(row.get("sample_id", -1)),

        "optimizer_success": bool(result.success),
        "target_success": target_success,
        "optimization_status": int(result.status),
        "optimization_message": str(result.message),
        "nfev": int(result.nfev),
        "least_squares_cost": float(result.cost),
        "optimization_runtime_seconds": float(optimization_runtime_seconds),
        "optimization_timed_out": False,

        "ballistic_proxy_cost": float(row.get("proxy_cost", np.nan)),
        "ballistic_position_error_norm_km": float(row.get("final_position_error_norm_km", np.nan)),
        "ballistic_velocity_error_norm_mps": float(row.get("final_velocity_error_norm_mps", np.nan)),

        "initial_x_geo_j2000_km": float(state0[0]),
        "initial_y_geo_j2000_km": float(state0[1]),
        "initial_z_geo_j2000_km": float(state0[2]),
        "initial_vx_geo_j2000_kms": float(state0[3]),
        "initial_vy_geo_j2000_kms": float(state0[4]),
        "initial_vz_geo_j2000_kms": float(state0[5]),
        "initial_C3_km2_s2": float(compute_C3_geo(state0)),

        "final_x_geo_j2000_km": float(final_state[0]),
        "final_y_geo_j2000_km": float(final_state[1]),
        "final_z_geo_j2000_km": float(final_state[2]),
        "final_vx_geo_j2000_kms": float(final_state[3]),
        "final_vy_geo_j2000_kms": float(final_state[4]),
        "final_vz_geo_j2000_kms": float(final_state[5]),

        "final_position_error_norm_km": final_pos_err,
        "final_velocity_error_norm_mps": final_vel_err_mps,
        "final_position_error_x_km": float(dr_final[0]),
        "final_position_error_y_km": float(dr_final[1]),
        "final_position_error_z_km": float(dr_final[2]),
        "final_velocity_error_x_mps": float(1000.0 * dv_final[0]),
        "final_velocity_error_y_mps": float(1000.0 * dv_final[1]),
        "final_velocity_error_z_mps": float(1000.0 * dv_final[2]),

        "dv_total_mps": dv_total_mps,
    }

    # Preserve useful dispersion metadata if present.
    for col in row.index:
        if str(col).startswith("dispersion_") or str(col).startswith("dpos_") or str(col).startswith("dvel_"):
            record[col] = row[col]

    # Preserve release geometry if present.
    for col in [
        "r0_mag_km",
        "C3_km2_s2",
        "phi_r_deg",
        "phi_v_deg",
        "out_of_plane_r_deg",
        "out_of_plane_v_deg",
        "nominal_source_sample_id",
    ]:
        if col in row.index:
            record[col] = row[col]

    for m, t_burn, dv_k, dv_mag_mps in zip(maneuvers, maneuver_times, dv_list, dv_mags_mps):
        name = m["name"]

        record[f"{name}_fraction"] = float(m["fraction"])
        record[f"{name}_jdtdb"] = float(t_burn)
        record[f"{name}_utc"] = jdtdb_to_utc(t_burn)

        record[f"{name}_dv_x_kms"] = float(dv_k[0])
        record[f"{name}_dv_y_kms"] = float(dv_k[1])
        record[f"{name}_dv_z_kms"] = float(dv_k[2])
        record[f"{name}_dv_mag_mps"] = float(dv_mag_mps)

    aux = {
        "state0": state0,
        "maneuvers": maneuvers,
        "maneuver_times": maneuver_times,
        "dv_list": dv_list,
        "event_records": event_records,
        "result": result,
    }

    return record, aux


# =============================================================================
# Staged trajectory output and visualization
# =============================================================================

def propagate_staged_trajectory(
    propagator,
    state0,
    t_depart_jdtdb,
    t_target_jdtdb,
    maneuvers,
    maneuver_times,
    dv_list,
    apply_count,
    config,
):
    step_days = float(config["propagation"]["step_days"])

    all_times = []
    all_states = []

    state = np.asarray(state0, dtype=float).copy()
    current_t = float(t_depart_jdtdb)

    for k, t_burn in enumerate(maneuver_times):
        t_burn = float(t_burn)

        if t_burn > current_t:
            t_grid = build_time_grid(current_t, t_burn, step_days)
            X = propagator.propagate(state, current_t, t_grid)

            if len(all_times) > 0:
                t_grid = t_grid[1:]
                X = X[1:]

            all_times.append(t_grid)
            all_states.append(X)

            state = X[-1].copy()
            current_t = t_burn

        if k < apply_count:
            state[3:] += dv_list[k]

    if current_t < float(t_target_jdtdb):
        t_grid = build_time_grid(current_t, t_target_jdtdb, step_days)
        X = propagator.propagate(state, current_t, t_grid)

        if len(all_times) > 0:
            t_grid = t_grid[1:]
            X = X[1:]

        all_times.append(t_grid)
        all_states.append(X)

    if len(all_times) == 0:
        return np.asarray([t_depart_jdtdb]), state0.reshape(1, 6)

    return np.hstack(all_times), np.vstack(all_states)


def trajectory_to_dataframe(t_grid, X, t_depart_jdtdb):
    r = X[:, :3]
    v = X[:, 3:]

    r_norm = np.linalg.norm(r, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    C3 = v_norm**2 - 2.0 * MU_EARTH_KM3_S2 / r_norm

    return pd.DataFrame({
        "jdtdb": t_grid,
        "utc": [jdtdb_to_utc(t) for t in t_grid],
        "t_days_since_departure": t_grid - float(t_depart_jdtdb),

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


def save_staged_trajectories_for_case(
    config,
    propagator,
    record,
    basis,
    t_depart_jdtdb,
    t_target_jdtdb,
    case_label,
):
    out_dir = Path(config["four_burn_optimization"]["output_dir"])
    traj_dir = out_dir / "representative_trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    state0 = get_initial_state_from_row(pd.Series(record), basis=basis)

    maneuvers = get_enabled_maneuvers(config)
    maneuver_times = maneuver_times_from_sequence(maneuvers, t_depart_jdtdb, t_target_jdtdb)

    dv_list = []
    for m in maneuvers:
        name = m["name"]
        dv_list.append(np.array([
            record[f"{name}_dv_x_kms"],
            record[f"{name}_dv_y_kms"],
            record[f"{name}_dv_z_kms"],
        ], dtype=float))

    labels = [("ballistic", 0)]
    for k in range(1, len(maneuvers) + 1):
        active_names = [m["name"] for m in maneuvers[:k]]
        labels.append(("_plus_".join(active_names), k))

    saved = {}

    for label, apply_count in labels:
        t_grid, X = propagate_staged_trajectory(
            propagator,
            state0,
            t_depart_jdtdb,
            t_target_jdtdb,
            maneuvers,
            maneuver_times,
            dv_list,
            apply_count,
            config,
        )

        df = trajectory_to_dataframe(t_grid, X, t_depart_jdtdb)

        path = traj_dir / f"{case_label}_{label}_trajectory.csv"
        df.to_csv(path, index=False)
        saved[label] = str(path)

    maneuver_rows = []
    for m, t_burn, dv in zip(maneuvers, maneuver_times, dv_list):
        maneuver_rows.append({
            "name": m["name"],
            "label": m.get("label", ""),
            "fraction": float(m["fraction"]),
            "jdtdb": float(t_burn),
            "utc": jdtdb_to_utc(t_burn),
            "dv_x_kms": float(dv[0]),
            "dv_y_kms": float(dv[1]),
            "dv_z_kms": float(dv[2]),
            "dv_mag_mps": float(1000.0 * np.linalg.norm(dv)),
        })

    maneuver_path = traj_dir / f"{case_label}_maneuvers.csv"
    pd.DataFrame(maneuver_rows).to_csv(maneuver_path, index=False)
    saved["maneuver_csv"] = str(maneuver_path)

    return saved


def set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    max_range = max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0]),
    )

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range / 2.0, x_mid + max_range / 2.0])
    ax.set_ylim3d([y_mid - max_range / 2.0, y_mid + max_range / 2.0])
    ax.set_zlim3d([z_mid - max_range / 2.0, z_mid + max_range / 2.0])


def plot_staged_trajectories(config, target_state, staged_paths, case_label):
    viz = config.get("visualization", {}).get("four_burn_representative_trajectories", {})
    if not bool(viz.get("save_staged_plots", True)) and not bool(viz.get("show_staged_plots", False)):
        return None

    out_dir = Path(config["four_burn_optimization"]["output_dir"])
    plot_dir = out_dir / "representative_trajectories"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    all_points = [
        np.zeros((1, 3)),
        np.asarray(target_state[:3]).reshape(1, 3),
    ]

    for label, path in staged_paths.items():
        if label == "maneuver_csv":
            continue

        df = pd.read_csv(path)
        r = df[["x_geo_j2000_km", "y_geo_j2000_km", "z_geo_j2000_km"]].to_numpy()
        all_points.append(r)

        pretty = label.replace("_plus_", " + ").replace("_", " ")
        ax.plot(r[:, 0], r[:, 1], r[:, 2], label=pretty)
        ax.scatter([r[-1, 0]], [r[-1, 1]], [r[-1, 2]], marker="x", s=40)

    ax.scatter([0.0], [0.0], [0.0], s=80, label="Earth")
    ax.scatter([target_state[0]], [target_state[1]], [target_state[2]], marker="*", s=100, label="Target")

    pts = np.vstack(all_points)
    lim = 1.1 * np.max(np.abs(pts))

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    ax.set_xlabel("GEO J2000 X [km]")
    ax.set_ylabel("GEO J2000 Y [km]")
    ax.set_zlabel("GEO J2000 Z [km]")
    ax.set_title(f"Staged four-burn trajectory: {case_label}")
    ax.legend(fontsize=7)

    set_axes_equal_3d(ax)
    plt.tight_layout()

    plot_path = plot_dir / f"{case_label}_staged_trajectories.png"

    if bool(viz.get("save_staged_plots", True)):
        fig.savefig(plot_path, dpi=300)

    if bool(viz.get("show_staged_plots", False)):
        plt.show()

    plt.close(fig)

    return str(plot_path)


def choose_representatives(opt_summary_df, config):
    viz = config.get("visualization", {}).get("four_burn_representative_trajectories", {})

    ok = opt_summary_df[
        (opt_summary_df["optimizer_success"] == True)
        & (opt_summary_df["target_success"] == True)
    ].copy()

    if ok.empty:
        ok = opt_summary_df[opt_summary_df["optimizer_success"] == True].copy()

    ok = ok.dropna(subset=["dv_total_mps"])

    if ok.empty:
        return {}

    ok = ok.sort_values("dv_total_mps").reset_index(drop=True)

    reps = {}

    if bool(viz.get("include_best_dv", True)):
        reps["best_dv"] = ok.iloc[0].to_dict()

    if bool(viz.get("include_median_dv", True)):
        median_dv = ok["dv_total_mps"].median()
        idx = int(np.argmin(np.abs(ok["dv_total_mps"].to_numpy() - median_dv)))
        reps["median_dv"] = ok.iloc[idx].to_dict()

    if bool(viz.get("include_worst_dv", False)):
        reps["worst_dv"] = ok.iloc[-1].to_dict()

    return reps



# =============================================================================
# MPI / runtime helpers
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI four-burn TCM optimization with optional dispersion."
    )
    parser.add_argument(
        "config",
        help="Path to MPI YAML config file.",
    )
    return parser.parse_args()


def should_save_and_stop(config, run_start_time):
    """
    Return True once the run is inside the configured walltime save buffer.

    This is intentionally checked only before starting a new optimization case.
    A case already in progress is allowed to finish.
    """
    runtime_cfg = config.get("runtime", {})
    walltime_hours = runtime_cfg.get("walltime_hours", None)

    if walltime_hours is None:
        return False

    walltime_s = 3600.0 * float(walltime_hours)
    buffer_s = 60.0 * float(runtime_cfg.get("save_before_walltime_minutes", 10.0))
    elapsed_s = time.time() - float(run_start_time)

    return elapsed_s >= walltime_s - buffer_s


def format_elapsed(seconds):
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def safe_case_sort_columns(df):
    cols = []
    for c in ["case_type", "sample_id", "case_id"]:
        if c in df.columns:
            cols.append(c)
    return cols


def make_failed_record(row, case_id, exception, runtime_seconds=None):
    timed_out = isinstance(exception, OptimizationTimeoutError)
    return {
        "case_id": str(case_id),
        "case_type": str(row.get("case_type", "unknown")),
        "sample_id": int(row.get("sample_id", -1)),
        "optimizer_success": False,
        "target_success": False,
        "optimization_status": -998 if timed_out else -999,
        "optimization_message": str(exception),
        "optimization_timed_out": bool(timed_out),
        "optimization_runtime_seconds": (
            float(runtime_seconds) if runtime_seconds is not None else np.nan
        ),
        "nfev": 0,
        "least_squares_cost": np.nan,
        "dv_total_mps": np.nan,
        "final_position_error_norm_km": np.nan,
        "final_velocity_error_norm_mps": np.nan,
    }


def save_rank_partial_csv(config, local_records, rank):
    """
    Save each rank's completed records to a unique CSV.

    This is mainly useful when the walltime buffer is reached and the run exits
    partially complete. It is not a full resume/checkpoint system.
    """
    if not bool(config.get("mpi", {}).get("save_rank_outputs", True)):
        return None

    out_dir = Path(config["four_burn_optimization"]["output_dir"])
    rank_dir = out_dir / "rank_outputs"
    rank_dir.mkdir(parents=True, exist_ok=True)

    final_path = rank_dir / f"four_burn_rank_{rank:04d}.csv"
    tmp_path = rank_dir / f"four_burn_rank_{rank:04d}.tmp.csv"

    pd.DataFrame(local_records).to_csv(tmp_path, index=False)
    tmp_path.replace(final_path)

    return final_path


def write_final_outputs(
    config,
    opt_summary_df,
    selected_df,
    nominal_row,
    target_state,
    t_depart_jdtdb,
    t_target_jdtdb,
    tof_days,
    maneuvers,
    input_mode,
    stopped_flags,
    rank_case_counts,
    rank_completed_counts,
    run_start_time,
    propagator,
    basis,
):
    out_dir = Path(config["four_burn_optimization"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "four_burn_optimized_sweep_summary.csv"
    metadata_path = out_dir / "four_burn_optimization_metadata.json"

    if bool(config.get("output", {}).get("save_four_burn_summary_csv", True)):
        opt_summary_df.to_csv(summary_path, index=False)

    if len(opt_summary_df) > 0:
        n_optimizer_success = int((opt_summary_df["optimizer_success"] == True).sum())
        n_target_success = int((opt_summary_df["target_success"] == True).sum())
    else:
        n_optimizer_success = 0
        n_target_success = 0

    n_fail = int(len(opt_summary_df) - n_optimizer_success)
    if "optimization_timed_out" in opt_summary_df.columns:
        n_optimization_timeouts = int((opt_summary_df["optimization_timed_out"] == True).sum())
    else:
        n_optimization_timeouts = 0
    stopped_for_walltime = bool(any(stopped_flags))

    metadata = {
        "input_mode": input_mode,
        "partial_run": stopped_for_walltime,
        "stopped_for_walltime": stopped_for_walltime,
        "n_selected_cases": int(len(selected_df)),
        "n_completed_cases": int(len(opt_summary_df)),
        "n_optimizer_success": n_optimizer_success,
        "n_target_success": n_target_success,
        "n_fail": n_fail,
        "n_optimization_timeouts": n_optimization_timeouts,

        "rank_case_counts": [int(x) for x in rank_case_counts],
        "rank_completed_counts": [int(x) for x in rank_completed_counts],
        "rank_stopped_for_walltime": [bool(x) for x in stopped_flags],
        "elapsed_wallclock_seconds": float(time.time() - run_start_time),
        "elapsed_wallclock_hhmmss": format_elapsed(time.time() - run_start_time),

        "target_index": int(config["target"]["target_index"]),
        "target_time_jdtdb": float(t_target_jdtdb),
        "target_time_utc": jdtdb_to_utc(t_target_jdtdb),
        "departure_time_jdtdb": float(t_depart_jdtdb),
        "departure_time_utc": jdtdb_to_utc(t_depart_jdtdb),
        "tof_days": float(tof_days),

        "enabled_maneuvers": [
            {
                "name": m["name"],
                "label": m.get("label", ""),
                "fraction": float(m["fraction"]),
                "dv_component_bound_kms": float(m["dv_component_bound_kms"]),
                "dv_weight": float(m.get("dv_weight", 1.0)),
            }
            for m in maneuvers
        ],

        "target_state_geo_eme_km_kms": target_state.tolist(),
        "runtime": config.get("runtime", {}),
    }

    if nominal_row is not None:
        metadata["nominal_source_sample_id"] = str(nominal_row.get("sample_id", ""))
        metadata["nominal_source_dv_total_mps"] = float(nominal_row.get("dv_total_mps", np.nan))
        metadata["dispersion"] = config.get("dispersion", {})

    representative_info = {}

    if (
        len(opt_summary_df) > 0
        and bool(config.get("output", {}).get("save_representative_trajectories", True))
    ):
        reps = choose_representatives(opt_summary_df, config)

        for rep_name, rep_record in reps.items():
            safe_case_id = str(rep_record["case_id"]).replace("/", "_").replace("\\", "_")
            case_label = f"{rep_name}_{safe_case_id}"

            staged_paths = save_staged_trajectories_for_case(
                config=config,
                propagator=propagator,
                record=rep_record,
                basis=basis,
                t_depart_jdtdb=t_depart_jdtdb,
                t_target_jdtdb=t_target_jdtdb,
                case_label=case_label,
            )

            representative_info[rep_name] = {
                "case_id": str(rep_record["case_id"]),
                "sample_id": int(rep_record.get("sample_id", -1)),
                "dv_total_mps": float(rep_record["dv_total_mps"]),
                "final_position_error_norm_km": float(rep_record["final_position_error_norm_km"]),
                "final_velocity_error_norm_mps": float(rep_record["final_velocity_error_norm_mps"]),
                "staged_paths": staged_paths,
            }

    metadata["representative_trajectories"] = representative_info

    if bool(config.get("output", {}).get("save_four_burn_metadata_json", True)):
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return summary_path, metadata_path, n_optimizer_success, n_target_success, n_fail, n_optimization_timeouts


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    config = load_config(args.config)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    run_start_time = time.time()

    if not bool(config.get("four_burn_optimization", {}).get("enabled", True)):
        if rank == 0:
            print("four_burn_optimization.enabled is false. Exiting.")
        return

    out_dir = Path(config["four_burn_optimization"]["output_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    load_spice_kernels(config)

    target_time_utc, target_state, target_row = load_lpf_target_state(config)

    tof_days = float(config["epoch"]["tof_days"])
    t_target_jdtdb = utc_to_jdtdb(target_time_utc)
    t_depart_jdtdb = t_target_jdtdb - tof_days

    maneuvers = get_enabled_maneuvers(config)
    input_mode = config["four_burn_optimization"].get("input_mode", "ballistic_summary")

    if rank == 0:
        print("MPI four-burn optimization")
        print("--------------------------")
        print(f"MPI ranks:        {size}")
        print(f"Input mode:       {input_mode}")
        print(f"Target index:     {config['target']['target_index']}")
        print(f"Target UTC:       {jdtdb_to_utc(t_target_jdtdb)}")
        print(f"Departure UTC:    {jdtdb_to_utc(t_depart_jdtdb)}")
        print(f"TOF days:         {tof_days:.6f}")
        print(f"Enabled burns:    {[m['name'] for m in maneuvers]}")
        print(f"Variables:        {3 * len(maneuvers)}")

        runtime_cfg = config.get("runtime", {})
        if runtime_cfg.get("walltime_hours", None) is not None:
            print(
                "Walltime save:   "
                f"{runtime_cfg['walltime_hours']} h walltime, "
                f"save/stop {runtime_cfg.get('save_before_walltime_minutes', 10.0)} min before end"
            )
        if runtime_cfg.get("max_optimization_minutes", None) is not None:
            print(
                "Case timeout:    "
                f"{runtime_cfg.get('max_optimization_minutes')} min per optimization case"
            )
        print()

        selected_df, nominal_row = build_input_cases(config)
    else:
        selected_df = None
        nominal_row = None

    selected_df, nominal_row = comm.bcast((selected_df, nominal_row), root=0)

    if selected_df.empty:
        raise RuntimeError("No cases selected for four-burn optimization.")

    local_df = selected_df.iloc[rank::size].copy()

    print(
        f"[rank {rank:04d}/{size:04d}] assigned {len(local_df)} cases",
        flush=True,
    )

    basis = get_earth_moon_basis(t_depart_jdtdb)
    propagator = build_nbody_propagator(config)

    progress_every = int(config.get("mpi", {}).get("progress_every", 1))
    local_records = []
    stopped_for_walltime = False

    for i_local, (_, row) in enumerate(local_df.iterrows()):
        if should_save_and_stop(config, run_start_time):
            print(
                f"[rank {rank:04d}] Walltime buffer reached. "
                f"Saving {len(local_records)} completed local cases and exiting.",
                flush=True,
            )
            stopped_for_walltime = True
            break

        case_id = str(row.get("case_id", row.get("sample_id", i_local)))

        if progress_every > 0 and (i_local == 0 or (i_local + 1) % progress_every == 0):
            print(
                f"[rank {rank:04d}] Optimizing local case "
                f"{i_local + 1}/{len(local_df)}: {case_id}",
                flush=True,
            )

        case_wallclock_start = time.time()

        try:
            record, aux = optimize_four_burn_for_case(
                row=row,
                basis=basis,
                propagator=propagator,
                target_state=target_state,
                t_depart_jdtdb=t_depart_jdtdb,
                t_target_jdtdb=t_target_jdtdb,
                config=config,
            )

        except Exception as e:
            record = make_failed_record(
                row,
                case_id,
                e,
                runtime_seconds=time.time() - case_wallclock_start,
            )

        local_records.append(record)

    rank_csv = save_rank_partial_csv(config, local_records, rank)
    if rank_csv is not None:
        print(f"[rank {rank:04d}] wrote {rank_csv}", flush=True)

    stopped_flags = comm.gather(stopped_for_walltime, root=0)
    rank_case_counts = comm.gather(len(local_df), root=0)
    rank_completed_counts = comm.gather(len(local_records), root=0)
    gathered = comm.gather(local_records, root=0)

    if rank != 0:
        return

    records = [rec for block in gathered for rec in block]
    opt_summary_df = pd.DataFrame(records)

    sort_cols = safe_case_sort_columns(opt_summary_df)
    if sort_cols:
        opt_summary_df = opt_summary_df.sort_values(sort_cols).reset_index(drop=True)

    (
        summary_path,
        metadata_path,
        n_optimizer_success,
        n_target_success,
        n_fail,
        n_optimization_timeouts,
    ) = write_final_outputs(
        config=config,
        opt_summary_df=opt_summary_df,
        selected_df=selected_df,
        nominal_row=nominal_row,
        target_state=target_state,
        t_depart_jdtdb=t_depart_jdtdb,
        t_target_jdtdb=t_target_jdtdb,
        tof_days=tof_days,
        maneuvers=maneuvers,
        input_mode=input_mode,
        stopped_flags=stopped_flags,
        rank_case_counts=rank_case_counts,
        rank_completed_counts=rank_completed_counts,
        run_start_time=run_start_time,
        propagator=propagator,
        basis=basis,
    )

    print()
    print("MPI four-burn optimization complete")
    print("-----------------------------------")
    print(f"Summary CSV:   {summary_path}")
    print(f"Metadata JSON: {metadata_path}")
    print(f"Completed cases:      {len(opt_summary_df)}/{len(selected_df)}")
    print(f"Stopped for walltime: {bool(any(stopped_flags))}")
    print(f"Optimizer successes:  {n_optimizer_success}")
    print(f"Target successes:     {n_target_success}")
    print(f"Failures:             {n_fail}")
    print(f"Optimization timeouts:{n_optimization_timeouts}")

    if len(opt_summary_df) > 0:
        ok = opt_summary_df[
            (opt_summary_df["optimizer_success"] == True)
            & (opt_summary_df["target_success"] == True)
        ].copy()

        if ok.empty:
            ok = opt_summary_df[opt_summary_df["optimizer_success"] == True].copy()

        ok = ok.dropna(subset=["dv_total_mps"])

        if not ok.empty:
            print()
            print("ΔV summary")
            print("----------")
            print(f"min total ΔV [m/s]:    {ok['dv_total_mps'].min():.3f}")
            print(f"median total ΔV [m/s]: {ok['dv_total_mps'].median():.3f}")
            print(f"mean total ΔV [m/s]:   {ok['dv_total_mps'].mean():.3f}")
            print(f"p95 total ΔV [m/s]:    {ok['dv_total_mps'].quantile(0.95):.3f}")
            print(f"max total ΔV [m/s]:    {ok['dv_total_mps'].max():.3f}")

            for m in maneuvers:
                col = f"{m['name']}_dv_mag_mps"
                if col in ok.columns:
                    print(f"median {m['name']} ΔV [m/s]: {ok[col].median():.3f}")


if __name__ == "__main__":
    main()
