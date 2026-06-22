"""
build_tli_departure_state.py

Purpose
-------
Build a constrained TLI-like cislunar release state using the Earth-Moon
basis at departure.

This script does NOT perform shooting or trajectory optimization yet.

It only:
    1. loads config,
    2. computes departure epoch,
    3. constructs Earth-Moon basis,
    4. builds a constrained high-energy geocentric state,
    5. saves outputs,
    6. optionally visualizes the geometry.

State convention
----------------
Output state is Earth-centered J2000/EME:

    [x, y, z, vx, vy, vz]

with:
    position in km
    velocity in km/s
"""

import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import spiceypy as spice

import matplotlib.pyplot as plt


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

    Example input:
        "2015-12-03 04:45:42.081531370"
    """
    et = spice.str2et(utc_string)
    jdtdb = spice.unitim(et, "ET", "JDTDB")
    return float(jdtdb)


def jdtdb_to_et(jdtdb):
    return float(spice.unitim(float(jdtdb), "JDTDB", "ET"))


def et_to_utc(et):
    return spice.et2utc(float(et), "ISOC", 6)


def jdtdb_to_utc(jdtdb):
    et = jdtdb_to_et(jdtdb)
    return et_to_utc(et)


# =============================================================================
# Earth-Moon basis
# =============================================================================

def get_moon_state_geo_j2000(t_jdtdb):
    """
    Moon state relative to Earth in J2000.

    Returns
    -------
    r_moon : np.ndarray, shape (3,)
        Earth-to-Moon position vector [km].

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
    Construct an Earth-Moon basis at departure.

    e1:
        Earth -> Moon direction.

    e3:
        Earth-Moon angular momentum direction.

    e2:
        Completes the right-handed frame. Approximately along lunar motion.

    Returns
    -------
    e1, e2, e3, r_moon, v_moon
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

    speed_sq = float(C3_km2_s2) + 2.0 * MU_EARTH_KM3_S2 / float(r_mag_km)

    if speed_sq <= 0.0:
        r_max = 2.0 * MU_EARTH_KM3_S2 / abs(float(C3_km2_s2))

        raise ValueError(
            "Invalid C3/radius combination.\n"
            f"  C3 = {C3_km2_s2:.6f} km^2/s^2\n"
            f"  r  = {r_mag_km:.6f} km\n"
            f"  v^2 = {speed_sq:.6f} km^2/s^2\n"
            f"For this negative C3, r must be less than about {r_max:.3f} km."
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
        near Earth-Moon line, with an in-plane offset phi_r and optional
        out-of-plane tilt.

    Velocity:
        magnitude fixed by C3, direction given by lunar lead angle phi_v and
        optional out-of-plane tilt.

    Returns
    -------
    state0 : np.ndarray, shape (6,)
        Earth-centered J2000 state [km, km/s].

    basis_info : dict
        Useful basis and geometry information.
    """

    inj = config["injection"]

    r0_mag = float(inj["r0_mag_km"])
    C3 = float(inj["C3_km2_s2"])

    phi_r = deg2rad(inj["phi_r_deg"])
    phi_v = deg2rad(inj["phi_v_deg"])

    out_r = deg2rad(inj.get("out_of_plane_r_deg", 0.0))
    out_v = deg2rad(inj.get("out_of_plane_v_deg", 0.0))

    e1, e2, e3, r_moon, v_moon = get_earth_moon_basis(t_depart_jdtdb)

    # -------------------------------------------------------------------------
    # Position direction
    #
    # phi_r = 0:
    #   position lies exactly along Earth -> Moon direction.
    #
    # positive phi_r:
    #   rotates toward approximate lunar-motion direction.
    #
    # out_r:
    #   small out-of-plane position tilt.
    # -------------------------------------------------------------------------
    r_hat_plane = np.cos(phi_r) * e1 + np.sin(phi_r) * e2
    r_hat = np.cos(out_r) * r_hat_plane + np.sin(out_r) * e3
    r_hat = unit(r_hat)

    # -------------------------------------------------------------------------
    # Velocity direction
    #
    # phi_v = 0:
    #   velocity points along Earth -> Moon direction.
    #
    # positive phi_v:
    #   velocity leads the Moon in its direction of motion.
    #
    # out_v:
    #   small out-of-plane velocity tilt.
    # -------------------------------------------------------------------------
    v_hat_plane = np.cos(phi_v) * e1 + np.sin(phi_v) * e2
    v_hat = np.cos(out_v) * v_hat_plane + np.sin(out_v) * e3
    v_hat = unit(v_hat)

    v0_mag = speed_from_C3(C3, r0_mag)

    r0 = r0_mag * r_hat
    v0 = v0_mag * v_hat

    state0 = np.hstack([r0, v0])

    # Diagnostics
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
    These checks are warnings only.
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
# Visualization
# =============================================================================

def plot_departure_geometry(config, state0, basis_info, output_dir):
    """
    Simple 3D geometry visualization.

    Shows:
        Earth at origin,
        Moon position,
        Earth-Moon line,
        injection position,
        injection velocity direction,
        Earth-Moon basis vectors.
    """

    viz = config.get("visualization", {})

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

    # Earth
    ax.scatter([0.0], [0.0], [0.0], s=80, label="Earth")

    # Moon
    ax.scatter(
        [r_moon[0]],
        [r_moon[1]],
        [r_moon[2]],
        s=60,
        label="Moon",
    )

    # Earth-Moon line
    ax.plot(
        [0.0, r_moon[0]],
        [0.0, r_moon[1]],
        [0.0, r_moon[2]],
        linestyle="--",
        label="Earth-Moon line",
    )

    # Injection position
    ax.scatter(
        [r0[0]],
        [r0[1]],
        [r0[2]],
        s=70,
        label="Release position",
    )

    # Velocity vector, scaled for display
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

    # Basis vectors near origin
    basis_scale = 0.15 * moon_dist

    ax.quiver(
        0.0, 0.0, 0.0,
        basis_scale * e1[0],
        basis_scale * e1[1],
        basis_scale * e1[2],
        length=1.0,
        normalize=False,
        label="e1: Earth → Moon",
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

    plt.tight_layout()

    if bool(viz.get("save_plot", True)):
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = viz.get("filename", "tli_departure_geometry.png")
        fig.savefig(output_dir / filename, dpi=300)

    if bool(viz.get("show_plot", False)):
        plt.show()

    plt.close(fig)


# =============================================================================
# Save outputs
# =============================================================================

def save_outputs(config, state0, basis_info, t_depart_jdtdb, t_target_jdtdb):
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
            }
        ])

        df.to_csv(output_dir / "tli_departure_state.csv", index=False)

    return output_dir


# =============================================================================
# Main
# =============================================================================

def main():
    config = load_config("transfer_tli_departure_config.yaml")

    load_spice_kernels(config)

    warnings = check_constraints(config)

    for w in warnings:
        print(f"WARNING: {w}")

    target_time_utc = config["epoch"]["target_time_utc"]
    tof_days = float(config["epoch"]["tof_days"])

    t_target_jdtdb = utc_to_jdtdb(target_time_utc)
    t_depart_jdtdb = t_target_jdtdb - tof_days

    print("Departure setup")
    print("----------------")
    print(f"Target UTC:    {jdtdb_to_utc(t_target_jdtdb)}")
    print(f"Departure UTC: {jdtdb_to_utc(t_depart_jdtdb)}")
    print(f"TOF days:      {tof_days:.6f}")
    print()

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

    output_dir = save_outputs(
        config=config,
        state0=state0,
        basis_info=basis_info,
        t_depart_jdtdb=t_depart_jdtdb,
        t_target_jdtdb=t_target_jdtdb,
    )

    plot_departure_geometry(
        config=config,
        state0=state0,
        basis_info=basis_info,
        output_dir=output_dir,
    )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()