"""
build_and_propagate_tli_departure.py

Purpose
-------
Build and propagate a constrained TLI-like cislunar release state.

This script performs:

    1. Load config and SPICE kernels.
    2. Load LPF target state at a specified index.
    3. Compute departure epoch = target epoch - fixed TOF.
    4. Build Earth-Moon basis at departure.
    5. Construct a TLI-like cislunar release state:
        - position near Earth-Moon line,
        - velocity with lunar lead angle,
        - speed fixed by prescribed C3.
    6. Optionally plot the initial departure geometry.
    7. Ballistically propagate the state with your NBodyPropagator.
    8. Save propagated trajectory and summary.
    9. Optionally plot the propagated trajectory.

State convention
----------------
All propagated states are Earth-centered J2000/EME:

    [x, y, z, vx, vy, vz]

with:
    position [km]
    velocity [km/s]

The LPF target state is read from GEO_EME columns, so no frame conversion is
needed at this stage.
"""

import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import spiceypy as spice

import matplotlib.pyplot as plt

from n_body_integrator import NBodyPropagator


SECONDS_PER_DAY = 86400.0
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


def deg2rad(x):
    return np.deg2rad(float(x))


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_spice_kernels(config):
    for kernel in config["spice"]["kernels"]:
        spice.furnsh(kernel)


def utc_to_jdtdb(utc_string):
    """
    Convert UTC string to JDTDB using SPICE.

    The LPF file example time format is:
        2015-12-03 04:45:42.081531370
    """
    et = spice.str2et(str(utc_string))
    jdtdb = spice.unitim(et, "ET", "JDTDB")
    return float(jdtdb)


def jdtdb_to_et(jdtdb):
    return float(spice.unitim(float(jdtdb), "JDTDB", "ET"))


def et_to_utc(et):
    return spice.et2utc(float(et), "ISOC", 6)


def jdtdb_to_utc(jdtdb):
    return et_to_utc(jdtdb_to_et(jdtdb))


# =============================================================================
# LPF target loading
# =============================================================================

def load_lpf_target_state(config):
    """
    Load the selected LPF target row.

    Returns
    -------
    target_time_utc : str
        Time from LPF file row.

    target_state : np.ndarray, shape (6,)
        GEO_EME/J2000 state [km, km/s].

    target_row : pd.Series
        Full row for debugging/export.
    """

    target_cfg = config["target"]

    lpf_path = config["files"]["lpf_orbit_csv"]
    df = pd.read_csv(lpf_path)

    idx = int(target_cfg["target_index"])
    cols = target_cfg["state_columns_geo_eme"]
    time_col = target_cfg["time_column"]

    if idx < 0 or idx >= len(df):
        raise IndexError(
            f"target_index={idx} outside LPF file length {len(df)}."
        )

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
    """
    Moon state relative to Earth in J2000.

    Returns
    -------
    r_moon : np.ndarray, shape (3,)
        Earth-to-Moon position [km].

    v_moon : np.ndarray, shape (3,)
        Moon velocity relative to Earth [km/s].
    """

    et = jdtdb_to_et(t_jdtdb)

    moon_state, _ = spice.spkgeo(
        301,        # Moon
        et,
        "J2000",
        399,        # Earth
    )

    moon_state = np.asarray(moon_state, dtype=float)

    r_moon = moon_state[:3]
    v_moon = moon_state[3:]

    return r_moon, v_moon


def get_earth_moon_basis(t_jdtdb):
    """
    Construct the Earth-Moon basis at departure.

    e1:
        Earth -> Moon direction.

    e3:
        Earth-Moon angular momentum direction.

    e2:
        Completes right-handed frame. Approximately lunar-motion direction.
    """

    r_moon, v_moon = get_moon_state_geo_j2000(t_jdtdb)

    e1 = unit(r_moon)
    e3 = unit(np.cross(r_moon, v_moon))
    e2 = unit(np.cross(e3, e1))

    return e1, e2, e3, r_moon, v_moon


# =============================================================================
# C3 and state construction
# =============================================================================

def speed_from_C3(C3_km2_s2, r_mag_km):
    """
    Compute two-body geocentric speed from C3 and radius.

    C3 = v^2 - 2 mu_E / r

    Therefore:
        v = sqrt(C3 + 2 mu_E / r)
    """

    C3 = float(C3_km2_s2)
    r_mag = float(r_mag_km)

    speed_sq = C3 + 2.0 * MU_EARTH_KM3_S2 / r_mag

    if speed_sq <= 0.0:
        if C3 < 0:
            r_max = 2.0 * MU_EARTH_KM3_S2 / abs(C3)
            extra = f"For this negative C3, r must be less than {r_max:.3f} km."
        else:
            extra = "Check C3 and r_mag."

        raise ValueError(
            "Invalid C3/radius combination.\n"
            f"  C3 = {C3:.6f} km^2/s^2\n"
            f"  r  = {r_mag:.6f} km\n"
            f"  v^2 = {speed_sq:.6f} km^2/s^2\n"
            f"  {extra}"
        )

    return np.sqrt(speed_sq)


def compute_C3_geo(state):
    state = np.asarray(state, dtype=float)

    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])

    return v**2 - 2.0 * MU_EARTH_KM3_S2 / r


def build_tli_like_state(config, t_depart_jdtdb):
    """
    Build a constrained TLI-like cislunar release state.

    Position:
        near Earth-Moon line, with in-plane angle phi_r and optional
        out-of-plane tilt.

    Velocity:
        magnitude fixed by C3, direction given by lunar lead angle phi_v and
        optional out-of-plane tilt.

    Returns
    -------
    state0 : np.ndarray, shape (6,)
        Earth-centered J2000 state [km, km/s].

    basis_info : dict
        Basis and geometry diagnostics.
    """

    inj = config["injection"]

    r0_mag = float(inj["r0_mag_km"])
    C3 = float(inj["C3_km2_s2"])

    phi_r = deg2rad(inj["phi_r_deg"])
    phi_v = deg2rad(inj["phi_v_deg"])

    out_r = deg2rad(inj.get("out_of_plane_r_deg", 0.0))
    out_v = deg2rad(inj.get("out_of_plane_v_deg", 0.0))

    e1, e2, e3, r_moon, v_moon = get_earth_moon_basis(t_depart_jdtdb)

    # Position direction:
    # phi_r = 0 means exactly along Earth -> Moon.
    r_hat_plane = np.cos(phi_r) * e1 + np.sin(phi_r) * e2
    r_hat = np.cos(out_r) * r_hat_plane + np.sin(out_r) * e3
    r_hat = unit(r_hat)

    # Velocity direction:
    # phi_v = 0 means along Earth -> Moon.
    # positive phi_v leads the Moon in the lunar-motion direction.
    v_hat_plane = np.cos(phi_v) * e1 + np.sin(phi_v) * e2
    v_hat = np.cos(out_v) * v_hat_plane + np.sin(out_v) * e3
    v_hat = unit(v_hat)

    v0_mag = speed_from_C3(C3, r0_mag)

    r0 = r0_mag * r_hat
    v0 = v0_mag * v_hat

    state0 = np.hstack([r0, v0])

    r_hat_actual = unit(r0)
    v_hat_actual = unit(v0)

    radial_velocity = float(np.dot(v0, r_hat_actual))

    angle_r_from_moon_line = np.rad2deg(
        np.arccos(np.clip(np.dot(r_hat_actual, e1), -1.0, 1.0))
    )

    angle_v_from_moon_line = np.rad2deg(
        np.arccos(np.clip(np.dot(v_hat_actual, e1), -1.0, 1.0))
    )

    out_of_plane_pos_angle = np.rad2deg(
        np.arcsin(np.clip(np.dot(r_hat_actual, e3), -1.0, 1.0))
    )

    out_of_plane_vel_angle = np.rad2deg(
        np.arcsin(np.clip(np.dot(v_hat_actual, e3), -1.0, 1.0))
    )

    basis_info = {
        "e1_earth_to_moon": e1.tolist(),
        "e2_lunar_motion_direction": e2.tolist(),
        "e3_earth_moon_angular_momentum": e3.tolist(),

        "moon_position_geo_j2000_km": r_moon.tolist(),
        "moon_velocity_geo_j2000_kms": v_moon.tolist(),
        "moon_distance_km": float(np.linalg.norm(r_moon)),

        "r0_mag_km": float(np.linalg.norm(r0)),
        "v0_mag_kms": float(np.linalg.norm(v0)),
        "C3_km2_s2": float(compute_C3_geo(state0)),

        "radial_velocity_kms": radial_velocity,

        "angle_position_from_earth_moon_line_deg": float(angle_r_from_moon_line),
        "angle_velocity_from_earth_moon_line_deg": float(angle_v_from_moon_line),
        "out_of_plane_position_angle_deg": float(out_of_plane_pos_angle),
        "out_of_plane_velocity_angle_deg": float(out_of_plane_vel_angle),
    }

    return state0, basis_info


# =============================================================================
# Config checks
# =============================================================================

def check_constraints(config):
    """
    Check user-defined geometry constraints.
    These are warnings only.
    """

    inj = config["injection"]
    cons = config.get("constraints", {})

    phi_r = float(inj["phi_r_deg"])
    phi_v = float(inj["phi_v_deg"])
    out_v = float(inj.get("out_of_plane_v_deg", 0.0))

    warnings = []

    max_abs_phi_r = float(cons.get("max_abs_phi_r_deg", 60.0))
    min_phi_v = float(cons.get("min_phi_v_deg", -30.0))
    max_phi_v = float(cons.get("max_phi_v_deg", 60.0))
    max_abs_out_v = float(cons.get("max_abs_out_of_plane_v_deg", 15.0))

    if abs(phi_r) > max_abs_phi_r:
        warnings.append(
            f"phi_r_deg={phi_r:.3f} exceeds recommended ±{max_abs_phi_r:.3f} deg."
        )

    if phi_v < min_phi_v or phi_v > max_phi_v:
        warnings.append(
            f"phi_v_deg={phi_v:.3f} outside recommended range "
            f"[{min_phi_v:.3f}, {max_phi_v:.3f}] deg."
        )

    if abs(out_v) > max_abs_out_v:
        warnings.append(
            f"out_of_plane_v_deg={out_v:.3f} exceeds recommended "
            f"±{max_abs_out_v:.3f} deg."
        )

    return warnings


# =============================================================================
# N-body propagator
# =============================================================================

def load_nbody_constants(config):
    constants_path = config["files"]["constants_yaml"]

    with open(constants_path, "r") as f:
        constants = yaml.safe_load(f)

    return constants


def build_nbody_propagator(config):
    """
    Build the Earth-centered J2000 n-body propagator.
    """

    constants = load_nbody_constants(config)
    prop_cfg = config["propagation"]

    propagator = NBodyPropagator(
        spice=spice,
        config=constants,
        bodies=tuple(prop_cfg["bodies"]),
        frame=prop_cfg.get("frame", "J2000"),
        origin=prop_cfg.get("origin", "399"),
        rtol=float(prop_cfg.get("rtol", 1e-10)),
        atol=float(prop_cfg.get("atol", 1e-12)),
        method=prop_cfg.get("method", "DOP853"),
    )

    return propagator


def build_time_grid(t0_jdtdb, tf_jdtdb, step_days):
    """
    Build a monotonically increasing JDTDB time grid.
    """

    t0 = float(t0_jdtdb)
    tf = float(tf_jdtdb)
    step = float(step_days)

    if step <= 0.0:
        raise ValueError("step_days must be positive.")

    if tf < t0:
        raise ValueError("Final epoch must be after initial epoch.")

    t_grid = np.arange(t0, tf, step)

    if t_grid.size == 0 or not np.isclose(t_grid[0], t0):
        t_grid = np.insert(t_grid, 0, t0)

    if t_grid[-1] < tf:
        t_grid = np.append(t_grid, tf)

    return t_grid


def propagate_ballistic_trajectory(
    propagator,
    state0,
    t_depart_jdtdb,
    t_target_jdtdb,
    config,
):
    """
    Propagate the initialized TLI-like state ballistically.
    """

    step_days = float(config["propagation"]["step_days"])
    debug_duration = config["propagation"].get("debug_duration_days", None)

    if debug_duration is not None:
        tf_jdtdb = float(t_depart_jdtdb) + float(debug_duration)
    else:
        tf_jdtdb = float(t_target_jdtdb)

    t_grid = build_time_grid(
        t0_jdtdb=t_depart_jdtdb,
        tf_jdtdb=tf_jdtdb,
        step_days=step_days,
    )

    X = propagator.propagate(
        x0_km=state0,
        t0_jdtdb=t_depart_jdtdb,
        t1_jdtdb=t_grid,
    )

    return t_grid, X


# =============================================================================
# Save outputs
# =============================================================================

def save_initial_state_outputs(
    config,
    state0,
    basis_info,
    t_depart_jdtdb,
    t_target_jdtdb,
    target_state,
):
    output_dir = Path(config["files"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_cfg = config.get("output", {})

    summary = {
        "departure_time_jdtdb": float(t_depart_jdtdb),
        "departure_time_utc": jdtdb_to_utc(t_depart_jdtdb),

        "target_time_jdtdb": float(t_target_jdtdb),
        "target_time_utc": jdtdb_to_utc(t_target_jdtdb),

        "tof_days": float(config["epoch"]["tof_days"]),

        "boundary_type": config["injection"]["boundary_type"],

        "state_geo_j2000_km_kms": {
            "x_km": float(state0[0]),
            "y_km": float(state0[1]),
            "z_km": float(state0[2]),
            "vx_kms": float(state0[3]),
            "vy_kms": float(state0[4]),
            "vz_kms": float(state0[5]),
        },

        "target_state_geo_eme_km_kms": {
            "x_km": float(target_state[0]),
            "y_km": float(target_state[1]),
            "z_km": float(target_state[2]),
            "vx_kms": float(target_state[3]),
            "vy_kms": float(target_state[4]),
            "vz_kms": float(target_state[5]),
        },

        "geometry": basis_info,
    }

    if bool(output_cfg.get("save_state_json", True)):
        with open(output_dir / "tli_departure_state_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    if bool(output_cfg.get("save_state_csv", True)):
        df = pd.DataFrame([
            {
                "departure_time_jdtdb": t_depart_jdtdb,
                "departure_time_utc": jdtdb_to_utc(t_depart_jdtdb),
                "target_time_jdtdb": t_target_jdtdb,
                "target_time_utc": jdtdb_to_utc(t_target_jdtdb),
                "tof_days": config["epoch"]["tof_days"],

                "x_geo_j2000_km": state0[0],
                "y_geo_j2000_km": state0[1],
                "z_geo_j2000_km": state0[2],
                "vx_geo_j2000_kms": state0[3],
                "vy_geo_j2000_kms": state0[4],
                "vz_geo_j2000_kms": state0[5],

                "r0_mag_km": basis_info["r0_mag_km"],
                "v0_mag_kms": basis_info["v0_mag_kms"],
                "C3_km2_s2": basis_info["C3_km2_s2"],
                "radial_velocity_kms": basis_info["radial_velocity_kms"],

                "moon_distance_km": basis_info["moon_distance_km"],
                "angle_position_from_earth_moon_line_deg":
                    basis_info["angle_position_from_earth_moon_line_deg"],
                "angle_velocity_from_earth_moon_line_deg":
                    basis_info["angle_velocity_from_earth_moon_line_deg"],
                "out_of_plane_position_angle_deg":
                    basis_info["out_of_plane_position_angle_deg"],
                "out_of_plane_velocity_angle_deg":
                    basis_info["out_of_plane_velocity_angle_deg"],

                "target_x_geo_eme_km": target_state[0],
                "target_y_geo_eme_km": target_state[1],
                "target_z_geo_eme_km": target_state[2],
                "target_vx_geo_eme_kms": target_state[3],
                "target_vy_geo_eme_kms": target_state[4],
                "target_vz_geo_eme_kms": target_state[5],
            }
        ])

        df.to_csv(output_dir / "tli_departure_state.csv", index=False)

    return output_dir


def save_propagated_trajectory(
    config,
    t_grid,
    X,
    target_state,
    t_depart_jdtdb,
    t_target_jdtdb,
):
    output_dir = Path(config["files"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_cfg = config.get("output", {})

    r = X[:, :3]
    v = X[:, 3:]

    r_norm = np.linalg.norm(r, axis=1)
    v_norm = np.linalg.norm(v, axis=1)

    C3 = v_norm**2 - 2.0 * MU_EARTH_KM3_S2 / r_norm

    final_state = X[-1]
    final_pos_error = final_state[:3] - target_state[:3]
    final_vel_error = final_state[3:] - target_state[3:]

    traj_path = output_dir / "ballistic_tli_propagation_geo_j2000.csv"

    if bool(output_cfg.get("save_trajectory_csv", True)):
        df = pd.DataFrame({
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

        df.to_csv(traj_path, index=False)

    summary = {
        "departure_time_jdtdb": float(t_depart_jdtdb),
        "departure_time_utc": jdtdb_to_utc(t_depart_jdtdb),
        "target_time_jdtdb": float(t_target_jdtdb),
        "target_time_utc": jdtdb_to_utc(t_target_jdtdb),
        "propagated_final_time_jdtdb": float(t_grid[-1]),
        "propagated_final_time_utc": jdtdb_to_utc(t_grid[-1]),

        "tof_days_configured": float(t_target_jdtdb - t_depart_jdtdb),
        "tof_days_propagated": float(t_grid[-1] - t_depart_jdtdb),

        "initial_state_geo_j2000_km_kms": X[0].tolist(),
        "final_state_geo_j2000_km_kms": final_state.tolist(),
        "target_state_geo_eme_km_kms": target_state.tolist(),

        "final_position_error_km": final_pos_error.tolist(),
        "final_velocity_error_kms": final_vel_error.tolist(),
        "final_position_error_norm_km": float(np.linalg.norm(final_pos_error)),
        "final_velocity_error_norm_mps": float(1000.0 * np.linalg.norm(final_vel_error)),

        "initial_C3_km2_s2": float(C3[0]),
        "final_C3_km2_s2": float(C3[-1]),

        "initial_geocentric_distance_km": float(r_norm[0]),
        "final_geocentric_distance_km": float(r_norm[-1]),
    }

    summary_path = output_dir / "ballistic_tli_propagation_summary.json"

    if bool(output_cfg.get("save_propagation_summary_json", True)):
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    return traj_path, summary_path, summary


# =============================================================================
# Visualization
# =============================================================================

def set_axes_equal_3d(ax):
    """
    Make 3D axes approximately equal scale.
    """

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


def plot_departure_geometry(config, state0, basis_info, output_dir):
    """
    Plot initial departure geometry only.

    Toggle:
        visualization.departure_geometry.enabled
    """

    viz = config.get("visualization", {}).get("departure_geometry", {})

    if not bool(viz.get("enabled", False)):
        return

    r0 = state0[:3]
    v0 = state0[3:]

    e1 = np.asarray(basis_info["e1_earth_to_moon"])
    e2 = np.asarray(basis_info["e2_lunar_motion_direction"])
    e3 = np.asarray(basis_info["e3_earth_moon_angular_momentum"])

    r_moon = np.asarray(basis_info["moon_position_geo_j2000_km"])

    moon_dist = np.linalg.norm(r_moon)
    r0_mag = np.linalg.norm(r0)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter([0.0], [0.0], [0.0], s=80, label="Earth")

    ax.scatter(
        [r_moon[0]],
        [r_moon[1]],
        [r_moon[2]],
        s=60,
        label="Moon at departure",
    )

    ax.plot(
        [0.0, r_moon[0]],
        [0.0, r_moon[1]],
        [0.0, r_moon[2]],
        linestyle="--",
        label="Earth-Moon line",
    )

    ax.scatter(
        [r0[0]],
        [r0[1]],
        [r0[2]],
        s=70,
        label="Release position",
    )

    v_hat = unit(v0)
    vel_scale = 0.20 * moon_dist

    ax.quiver(
        r0[0],
        r0[1],
        r0[2],
        vel_scale * v_hat[0],
        vel_scale * v_hat[1],
        vel_scale * v_hat[2],
        length=1.0,
        normalize=False,
        label="Release velocity direction",
    )

    basis_scale = 0.15 * moon_dist

    ax.quiver(
        0.0, 0.0, 0.0,
        basis_scale * e1[0],
        basis_scale * e1[1],
        basis_scale * e1[2],
        length=1.0,
        normalize=False,
        label="e1: Earth to Moon",
    )

    ax.quiver(
        0.0, 0.0, 0.0,
        basis_scale * e2[0],
        basis_scale * e2[1],
        basis_scale * e2[2],
        length=1.0,
        normalize=False,
        label="e2: lunar motion",
    )

    ax.quiver(
        0.0, 0.0, 0.0,
        basis_scale * e3[0],
        basis_scale * e3[1],
        basis_scale * e3[2],
        length=1.0,
        normalize=False,
        label="e3: EM angular momentum",
    )

    lim = 1.1 * max(moon_dist, r0_mag)

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    ax.set_xlabel("GEO J2000 X [km]")
    ax.set_ylabel("GEO J2000 Y [km]")
    ax.set_zlabel("GEO J2000 Z [km]")

    title = (
        "TLI-like cislunar release geometry\n"
        f"r0={r0_mag:.0f} km, "
        f"C3={basis_info['C3_km2_s2']:.3f} km²/s²"
    )
    ax.set_title(title)

    ax.legend(loc="upper left", fontsize=8)

    set_axes_equal_3d(ax)
    plt.tight_layout()

    if bool(viz.get("save_plot", True)):
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = viz.get("filename", "tli_departure_geometry.png")
        fig.savefig(output_dir / filename, dpi=300)

    if bool(viz.get("show_plot", False)):
        plt.show()

    plt.close(fig)


def plot_ballistic_trajectory(config, t_grid, X, target_state, basis_info):
    """
    Plot propagated trajectory.

    Toggle:
        visualization.propagated_trajectory.enabled
    """

    viz = config.get("visualization", {}).get("propagated_trajectory", {})

    if not bool(viz.get("enabled", False)):
        return

    output_dir = Path(config["files"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    r = X[:, :3]

    r_moon_depart = np.asarray(basis_info["moon_position_geo_j2000_km"])
    r0 = X[0, :3]
    rf = X[-1, :3]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        r[:, 0],
        r[:, 1],
        r[:, 2],
        label="Ballistic propagated trajectory",
    )

    ax.scatter([0.0], [0.0], [0.0], s=80, label="Earth")

    ax.scatter(
        [r_moon_depart[0]],
        [r_moon_depart[1]],
        [r_moon_depart[2]],
        s=60,
        label="Moon at departure",
    )

    ax.plot(
        [0.0, r_moon_depart[0]],
        [0.0, r_moon_depart[1]],
        [0.0, r_moon_depart[2]],
        linestyle="--",
        label="Earth-Moon line at departure",
    )

    ax.scatter(
        [r0[0]],
        [r0[1]],
        [r0[2]],
        s=60,
        label="Departure state",
    )

    ax.scatter(
        [rf[0]],
        [rf[1]],
        [rf[2]],
        s=60,
        label="Final propagated state",
    )

    ax.scatter(
        [target_state[0]],
        [target_state[1]],
        [target_state[2]],
        s=70,
        marker="x",
        label="LPF target state",
    )

    all_points = np.vstack([
        r,
        np.zeros((1, 3)),
        r_moon_depart.reshape(1, 3),
        target_state[:3].reshape(1, 3),
    ])

    max_abs = np.max(np.abs(all_points))
    lim = 1.1 * max_abs

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    ax.set_xlabel("GEO J2000 X [km]")
    ax.set_ylabel("GEO J2000 Y [km]")
    ax.set_zlabel("GEO J2000 Z [km]")

    ax.set_title("Ballistic propagation from TLI-like cislunar release state")
    ax.legend(fontsize=8)

    set_axes_equal_3d(ax)
    plt.tight_layout()

    if bool(viz.get("save_plot", True)):
        filename = viz.get("filename", "ballistic_tli_propagation.png")
        fig.savefig(output_dir / filename, dpi=300)

    if bool(viz.get("show_plot", False)):
        plt.show()

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    config = load_config("transfer_tli_departure_config.yaml")

    load_spice_kernels(config)

    warnings = check_constraints(config)
    for w in warnings:
        print(f"WARNING: {w}")

    # -------------------------------------------------------------------------
    # Load LPF target row
    # -------------------------------------------------------------------------
    target_time_utc, target_state, target_row = load_lpf_target_state(config)

    tof_days = float(config["epoch"]["tof_days"])

    t_target_jdtdb = utc_to_jdtdb(target_time_utc)
    t_depart_jdtdb = t_target_jdtdb - tof_days

    print("Target/departure setup")
    print("----------------------")
    print(f"LPF target index: {config['target']['target_index']}")
    print(f"Target UTC:       {jdtdb_to_utc(t_target_jdtdb)}")
    print(f"Departure UTC:    {jdtdb_to_utc(t_depart_jdtdb)}")
    print(f"TOF days:         {tof_days:.6f}")
    print()

    # -------------------------------------------------------------------------
    # Build initial TLI-like cislunar state
    # -------------------------------------------------------------------------
    state0, basis_info = build_tli_like_state(
        config=config,
        t_depart_jdtdb=t_depart_jdtdb,
    )

    print("Constructed TLI-like cislunar release state")
    print("-------------------------------------------")
    print(f"r0 [km]      = {state0[:3]}")
    print(f"v0 [km/s]    = {state0[3:]}")
    print(f"|r0| [km]    = {basis_info['r0_mag_km']:.6f}")
    print(f"|v0| [km/s]  = {basis_info['v0_mag_kms']:.9f}")
    print(f"C3 [km2/s2]  = {basis_info['C3_km2_s2']:.9f}")
    print(f"vr [km/s]    = {basis_info['radial_velocity_kms']:.9f}")
    print()

    output_dir = save_initial_state_outputs(
        config=config,
        state0=state0,
        basis_info=basis_info,
        t_depart_jdtdb=t_depart_jdtdb,
        t_target_jdtdb=t_target_jdtdb,
        target_state=target_state,
    )

    plot_departure_geometry(
        config=config,
        state0=state0,
        basis_info=basis_info,
        output_dir=output_dir,
    )

    # -------------------------------------------------------------------------
    # Propagate ballistically
    # -------------------------------------------------------------------------
    if bool(config.get("propagation", {}).get("enabled", True)):
        propagator = build_nbody_propagator(config)

        t_grid, X = propagate_ballistic_trajectory(
            propagator=propagator,
            state0=state0,
            t_depart_jdtdb=t_depart_jdtdb,
            t_target_jdtdb=t_target_jdtdb,
            config=config,
        )

        traj_path, summary_path, summary = save_propagated_trajectory(
            config=config,
            t_grid=t_grid,
            X=X,
            target_state=target_state,
            t_depart_jdtdb=t_depart_jdtdb,
            t_target_jdtdb=t_target_jdtdb,
        )

        plot_ballistic_trajectory(
            config=config,
            t_grid=t_grid,
            X=X,
            target_state=target_state,
            basis_info=basis_info,
        )

        print("Ballistic propagation complete")
        print("------------------------------")
        print(f"Trajectory saved to: {traj_path}")
        print(f"Summary saved to:    {summary_path}")
        print(
            "Final position error norm [km]: "
            f"{summary['final_position_error_norm_km']:.6e}"
        )
        print(
            "Final velocity error norm [m/s]: "
            f"{summary['final_velocity_error_norm_mps']:.6e}"
        )

    print()
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()