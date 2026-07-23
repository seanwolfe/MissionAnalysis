"""
MPI deployment-recovery dispersion analysis for one LPF reference trajectory.

Purpose
-------
The mothership is assumed to follow the time-tagged LPF reference trajectory and
release one microsatellite at the configured deployment index. Only the
separation azimuth, separation elevation, and separation-speed magnitude are
dispersed.

For every dispersion realization:

    LPF mothership state at deployment
        -> apply dispersed separation impulse
        -> ballistic commissioning coast
        -> burn 1 immediately after commissioning
        -> recovery coast for a candidate duration
        -> burn 2 at capture
        -> match the LPF position and velocity at that same capture epoch

A grid of recovery-transfer durations is evaluated. For each fixed duration,
scipy.optimize.least_squares solves six burn variables against six normalized
terminal-state residuals. Among optimizer-successful durations, the solution
with the lowest total recovery Delta-V is retained.

MPI behaviour
-------------
Dispersion realizations are distributed by deterministic round robin. Each rank
can save partial summary and recovery-trial CSV files. Rank 0 writes the master
outputs, metadata, representative trajectories, and plots.

The Python deadline checks cannot interrupt NBodyPropagator while an individual
propagation call is executing internally.

Usage
-----
    mpiexec -n 8 python deployment_recovery_dispersion_mpi.py \
        deployment_recovery_dispersion_config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mpi4py.rc

mpi4py.rc.threads = False
from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as spice
import yaml
from scipy.optimize import least_squares

from n_body_integrator import NBodyPropagator
import utilities as util


SECONDS_PER_DAY = 86400.0
NANOSECONDS_PER_DAY = SECONDS_PER_DAY * 1.0e9


class SampleTimeoutError(RuntimeError):
    """Raised when one complete dispersion realization exceeds its deadline."""


@dataclass(frozen=True)
class ReferenceOrbit:
    dataframe: pd.DataFrame
    period_dataframe: pd.DataFrame
    time_column: str
    state_columns: list[str]
    start_index: int
    end_index: int
    deployment_index: int
    start_jdtdb: float
    end_jdtdb: float
    period_days: float
    deployment_jdtdb: float
    deployment_utc: str
    deployment_state_eme: np.ndarray
    deployment_state_secr: np.ndarray
    deployment_local_basis_secr: np.ndarray
    source_elapsed_days: np.ndarray
    source_states: np.ndarray


# =============================================================================
# Generic utilities
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPI microsatellite deployment-recovery dispersion analysis."
    )
    parser.add_argument("config", help="Path to the YAML configuration file.")
    return parser.parse_args()


def rank_print(rank: int, message: str) -> None:
    print(f"[rank {rank:04d}] {message}", flush=True)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration root must be a mapping.")
    return config


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    return bool(value)


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return vector / norm


def load_spice_kernels(config: dict[str, Any]) -> None:
    for kernel in config["spice"]["kernels"]:
        spice.furnsh(str(kernel))


def utc_to_jdtdb(utc_value: Any) -> float:
    timestamp = pd.to_datetime(str(utc_value), utc=True)
    utc_clean = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")
    et = spice.str2et(utc_clean)
    return float(spice.unitim(et, "ET", "JDTDB"))


def jdtdb_to_et(jdtdb: float) -> float:
    return float(spice.unitim(float(jdtdb), "JDTDB", "ET"))


def jdtdb_to_utc(jdtdb: float) -> str:
    return str(spice.et2utc(jdtdb_to_et(float(jdtdb)), "ISOC", 6))


def get_earth_heliocentric_eclip_state(jdtdb: float) -> np.ndarray:
    state, _ = spice.spkgeo(399, jdtdb_to_et(jdtdb), "ECLIPJ2000", 10)
    return np.asarray(state, dtype=float)


def state_eme_to_secr(state_eme: np.ndarray, jdtdb: float) -> np.ndarray:
    earth_state = get_earth_heliocentric_eclip_state(jdtdb)
    state_eclip = util.geo_eme_to_geo_eclip_generic(
        np.asarray(state_eme, dtype=float), hint=("state",)
    )
    return np.asarray(
        util.geo_eclip_to_geo_secr_generic(
            state_eclip,
            earth_state,
            obj_hint=("state",),
            earth_hint=("state",),
        ),
        dtype=float,
    )


def state_secr_to_eme(state_secr: np.ndarray, jdtdb: float) -> np.ndarray:
    earth_state = get_earth_heliocentric_eclip_state(jdtdb)
    state_eclip = util.geo_secr_to_geo_eclip_generic(
        np.asarray(state_secr, dtype=float),
        earth_state,
        obj_hint=("state",),
        earth_hint=("state",),
    )
    return np.asarray(
        util.geo_eclip_to_geo_eme_generic(state_eclip, hint=("state",)),
        dtype=float,
    )


def vector_secr_to_eme(vector_secr: np.ndarray, jdtdb: float) -> np.ndarray:
    dummy_state = np.hstack([np.zeros(3), np.asarray(vector_secr, dtype=float)])
    return state_secr_to_eme(dummy_state, jdtdb)[3:]


def build_local_quasi_halo_basis(state_secr: np.ndarray) -> np.ndarray:
    state_secr = np.asarray(state_secr, dtype=float)
    position = state_secr[:3]
    velocity = state_secr[3:]
    tangent = unit(velocity)
    normal = unit(np.cross(position, velocity))
    cross_track = unit(np.cross(normal, tangent))
    basis = np.column_stack([tangent, cross_track, normal])
    if np.linalg.det(basis) < 0.0:
        cross_track = -cross_track
        basis = np.column_stack([tangent, cross_track, normal])
    return basis


def local_spherical_velocity_secr(
    speed_mps: float,
    azimuth_rad: float,
    elevation_rad: float,
    basis_secr: np.ndarray,
) -> np.ndarray:
    local_components = np.array(
        [
            math.cos(elevation_rad) * math.cos(azimuth_rad),
            math.cos(elevation_rad) * math.sin(azimuth_rad),
            math.sin(elevation_rad),
        ],
        dtype=float,
    )
    return (float(speed_mps) / 1000.0) * (basis_secr @ local_components)


def build_time_grid(t0: float, tf: float, step_days: float) -> np.ndarray:
    if step_days <= 0.0:
        raise ValueError("output.trajectory_step_days must be positive.")
    grid = np.arange(float(t0), float(tf), float(step_days), dtype=float)
    if grid.size == 0 or not np.isclose(grid[0], t0):
        grid = np.insert(grid, 0, t0)
    if not np.isclose(grid[-1], tf):
        grid = np.append(grid, tf)
    else:
        grid[-1] = tf
    return np.asarray(grid, dtype=float)


def set_axes_equal_3d(ax: Any) -> None:
    limits = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    maximum_range = max(abs(upper - lower) for lower, upper in limits)
    mids = [0.5 * (lower + upper) for lower, upper in limits]
    half = 0.5 * maximum_range
    ax.set_xlim3d(mids[0] - half, mids[0] + half)
    ax.set_ylim3d(mids[1] - half, mids[1] + half)
    ax.set_zlim3d(mids[2] - half, mids[2] + half)


# =============================================================================
# Reference orbit and propagation
# =============================================================================


def load_reference_orbit(config: dict[str, Any]) -> ReferenceOrbit:
    ref_cfg = config["reference_orbit"]
    dep_cfg = config["deployment"]
    dataframe = pd.read_csv(config["files"]["lpf_orbit_csv"])

    time_column = str(ref_cfg["time_column"])
    state_columns = list(ref_cfg["state_columns_geo_eme"])
    missing = [
        column for column in [time_column, *state_columns] if column not in dataframe.columns
    ]
    if missing:
        raise KeyError(f"LPF reference CSV is missing columns: {missing}")

    start_index = int(ref_cfg["quasi_halo_start"])
    end_index = int(ref_cfg["quasi_halo_one_period_end"])
    deployment_index = int(dep_cfg["nominal_deployment_index"])

    if start_index < 0 or end_index >= len(dataframe) or end_index <= start_index:
        raise IndexError(
            f"Invalid period indices start={start_index}, end={end_index}, rows={len(dataframe)}."
        )
    if deployment_index < start_index or deployment_index > end_index:
        raise ValueError("The deployment index must lie inside the selected LPF period.")

    period_dataframe = dataframe.iloc[start_index : end_index + 1].copy()
    period_timestamps = pd.DatetimeIndex(
        pd.to_datetime(period_dataframe[time_column].astype(str), utc=True)
    )
    source_elapsed_days = (
        period_timestamps.asi8 - period_timestamps.asi8[0]
    ) / NANOSECONDS_PER_DAY
    source_states = period_dataframe[state_columns].to_numpy(dtype=float)

    start_jdtdb = utc_to_jdtdb(period_dataframe.iloc[0][time_column])
    end_jdtdb = utc_to_jdtdb(period_dataframe.iloc[-1][time_column])
    period_days = float(end_jdtdb - start_jdtdb)

    deployment_row = dataframe.iloc[deployment_index]
    deployment_jdtdb = utc_to_jdtdb(deployment_row[time_column])
    deployment_state_eme = deployment_row[state_columns].to_numpy(dtype=float)
    deployment_state_secr = state_eme_to_secr(deployment_state_eme, deployment_jdtdb)
    deployment_basis = build_local_quasi_halo_basis(deployment_state_secr)

    return ReferenceOrbit(
        dataframe=dataframe,
        period_dataframe=period_dataframe,
        time_column=time_column,
        state_columns=state_columns,
        start_index=start_index,
        end_index=end_index,
        deployment_index=deployment_index,
        start_jdtdb=float(start_jdtdb),
        end_jdtdb=float(end_jdtdb),
        period_days=period_days,
        deployment_jdtdb=float(deployment_jdtdb),
        deployment_utc=jdtdb_to_utc(deployment_jdtdb),
        deployment_state_eme=np.asarray(deployment_state_eme, dtype=float),
        deployment_state_secr=np.asarray(deployment_state_secr, dtype=float),
        deployment_local_basis_secr=np.asarray(deployment_basis, dtype=float),
        source_elapsed_days=np.asarray(source_elapsed_days, dtype=float),
        source_states=np.asarray(source_states, dtype=float),
    )


def interpolate_reference_states(
    reference: ReferenceOrbit, query_times_jdtdb: np.ndarray | list[float] | float
) -> np.ndarray:
    query = np.atleast_1d(np.asarray(query_times_jdtdb, dtype=float))
    query_elapsed = query - reference.start_jdtdb
    tolerance = 1.0e-9
    if query_elapsed.min() < reference.source_elapsed_days[0] - tolerance:
        raise ValueError("Requested reference interpolation precedes the selected period.")
    if query_elapsed.max() > reference.source_elapsed_days[-1] + tolerance:
        raise ValueError("Requested reference interpolation exceeds the selected period.")
    states = np.column_stack(
        [
            np.interp(query_elapsed, reference.source_elapsed_days, reference.source_states[:, i])
            for i in range(6)
        ]
    )
    return states[0] if np.asarray(query_times_jdtdb).ndim == 0 else states


def load_nbody_constants(config: dict[str, Any]) -> dict[str, Any]:
    with open(config["files"]["constants_yaml"], "r", encoding="utf-8") as stream:
        constants = yaml.safe_load(stream)
    if not isinstance(constants, dict):
        raise ValueError("The n-body constants YAML root must be a mapping.")
    return constants


def build_nbody_propagator(config: dict[str, Any]) -> NBodyPropagator:
    constants = load_nbody_constants(config)
    prop_cfg = config["propagation"]
    return NBodyPropagator(
        spice=spice,
        config=constants,
        bodies=tuple(prop_cfg["bodies"]),
        frame=prop_cfg.get("frame", "J2000"),
        origin=prop_cfg.get("origin", "399"),
        rtol=float(prop_cfg.get("rtol", 1.0e-10)),
        atol=float(prop_cfg.get("atol", 1.0e-12)),
        method=prop_cfg.get("method", "DOP853"),
    )


def propagate_final_state(
    propagator: NBodyPropagator,
    state0: np.ndarray,
    t0_jdtdb: float,
    tf_jdtdb: float,
) -> np.ndarray:
    if np.isclose(t0_jdtdb, tf_jdtdb, atol=1.0e-14, rtol=0.0):
        return np.asarray(state0, dtype=float).copy()
    state = propagator.propagate(
        x0_km=np.asarray(state0, dtype=float),
        t0_jdtdb=float(t0_jdtdb),
        t1_jdtdb=float(tf_jdtdb),
    )
    return np.asarray(state, dtype=float)


# =============================================================================
# Dispersion samples and recovery-time grid
# =============================================================================


def draw_errors(
    rng: np.random.Generator, block: dict[str, Any], count: int
) -> np.ndarray:
    distribution = str(block.get("distribution", "normal")).lower()
    if distribution == "normal":
        mean = float(block.get("mean", 0.0))
        sigma = float(block["sigma"])
        if sigma < 0.0:
            raise ValueError("Normal dispersion sigma cannot be negative.")
        values = rng.normal(mean, sigma, size=count)
        clip_sigma = block.get("clip_sigma")
        if clip_sigma is not None:
            width = abs(float(clip_sigma)) * sigma
            values = np.clip(values, mean - width, mean + width)
    elif distribution == "uniform":
        if "min" in block and "max" in block:
            lower = float(block["min"])
            upper = float(block["max"])
        else:
            half_range = abs(float(block["half_range"]))
            lower, upper = -half_range, half_range
        if upper < lower:
            raise ValueError("Uniform dispersion maximum is below its minimum.")
        values = rng.uniform(lower, upper, size=count)
    else:
        raise ValueError(f"Unknown dispersion distribution: {distribution}")
    return np.asarray(values, dtype=float)


def generate_dispersion_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    disp_cfg = config["dispersion"]
    sep_cfg = config["separation"]
    random_count = int(disp_cfg.get("n_random_samples", 0))
    include_nominal = as_bool(disp_cfg.get("include_nominal_case", True))
    if random_count < 0:
        raise ValueError("dispersion.n_random_samples cannot be negative.")

    rng = np.random.default_rng(int(disp_cfg.get("random_seed", 1)))
    az_errors = draw_errors(rng, disp_cfg["azimuth_error_deg"], random_count)
    el_errors = draw_errors(rng, disp_cfg["elevation_error_deg"], random_count)
    speed_errors = draw_errors(rng, disp_cfg["separation_speed_error_mps"], random_count)

    nominal_az = float(sep_cfg["nominal_azimuth_deg"])
    nominal_el = float(sep_cfg["nominal_elevation_deg"])
    nominal_speed = float(sep_cfg["nominal_speed_mps"])

    records: list[dict[str, Any]] = []
    if include_nominal:
        records.append(
            {
                "sample_id": 0,
                "case_id": "recovery_000000",
                "is_nominal_case": True,
                "azimuth_error_deg": 0.0,
                "elevation_error_deg": 0.0,
                "separation_speed_error_mps": 0.0,
                "separation_azimuth_deg": nominal_az,
                "separation_elevation_deg": nominal_el,
                "separation_speed_mps": nominal_speed,
            }
        )

    offset = 1 if include_nominal else 0
    for index in range(random_count):
        sample_id = offset + index
        azimuth = nominal_az + float(az_errors[index])
        # Wrap azimuth to [-180, 180).
        azimuth = ((azimuth + 180.0) % 360.0) - 180.0
        elevation = nominal_el + float(el_errors[index])
        speed = nominal_speed + float(speed_errors[index])
        records.append(
            {
                "sample_id": int(sample_id),
                "case_id": f"recovery_{sample_id:06d}",
                "is_nominal_case": False,
                "azimuth_error_deg": float(az_errors[index]),
                "elevation_error_deg": float(el_errors[index]),
                "separation_speed_error_mps": float(speed_errors[index]),
                "separation_azimuth_deg": float(azimuth),
                "separation_elevation_deg": float(elevation),
                "separation_speed_mps": float(speed),
            }
        )
    return records


def build_recovery_time_grid(config: dict[str, Any], reference: ReferenceOrbit) -> np.ndarray:
    recovery_cfg = config["recovery"]
    grid_cfg = recovery_cfg["transfer_time_grid"]
    commissioning = float(recovery_cfg["commissioning_duration_days"])
    maximum_available = reference.end_jdtdb - (
        reference.deployment_jdtdb + commissioning
    )
    if maximum_available <= 0.0:
        raise ValueError("Commissioning ends after the selected LPF period.")

    mode = str(grid_cfg.get("mode", "linspace")).lower()
    if mode == "explicit":
        values = np.asarray(grid_cfg["values_days"], dtype=float)
    elif mode == "linspace":
        minimum = float(grid_cfg["minimum_days"])
        configured_maximum = grid_cfg.get("maximum_days")
        maximum = (
            maximum_available
            if configured_maximum is None
            else float(configured_maximum)
        )
        count = int(grid_cfg["number_of_times"])
        if count < 1:
            raise ValueError("recovery.transfer_time_grid.number_of_times must be >= 1.")
        values = np.linspace(minimum, maximum, count)
    else:
        raise ValueError(f"Unknown recovery transfer-time grid mode: {mode}")

    values = np.unique(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any(values <= 0.0):
        raise ValueError("Every recovery transfer duration must be positive.")
    if values.max() > maximum_available + 1.0e-9:
        raise ValueError(
            "The recovery grid exceeds the available LPF reference interval after commissioning."
        )
    return values


def build_post_separation_state(
    sample: dict[str, Any], reference: ReferenceOrbit
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth_rad = math.radians(float(sample["separation_azimuth_deg"]))
    elevation_rad = math.radians(float(sample["separation_elevation_deg"]))
    separation_velocity_secr = local_spherical_velocity_secr(
        speed_mps=float(sample["separation_speed_mps"]),
        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,
        basis_secr=reference.deployment_local_basis_secr,
    )
    separation_velocity_eme = vector_secr_to_eme(
        separation_velocity_secr, reference.deployment_jdtdb
    )
    state0 = reference.deployment_state_eme.copy()
    state0[3:] += separation_velocity_eme
    return state0, separation_velocity_secr, separation_velocity_eme


# =============================================================================
# Six-variable two-burn recovery optimization
# =============================================================================


def optimizer_bounds_and_guess(
    config: dict[str, Any],
    state_commissioning_end: np.ndarray,
    t_burn1: float,
    capture_jdtdb: float,
    target_state: np.ndarray,
    propagator: NBodyPropagator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    opt_cfg = config["optimization"]
    burn1_bound = float(opt_cfg["first_burn_component_bound_mps"]) / 1000.0
    burn2_bound = float(opt_cfg["second_burn_component_bound_mps"]) / 1000.0
    lower = np.array([-burn1_bound] * 3 + [-burn2_bound] * 3, dtype=float)
    upper = np.array([burn1_bound] * 3 + [burn2_bound] * 3, dtype=float)

    mode = str(opt_cfg.get("initial_guess_mode", "ballistic_informed")).lower()
    if mode == "zero":
        guess = np.zeros(6, dtype=float)
    elif mode == "ballistic_informed":
        ballistic_final = propagate_final_state(
            propagator,
            state_commissioning_end,
            t_burn1,
            capture_jdtdb,
        )
        guess = np.zeros(6, dtype=float)
        guess[3:] = target_state[3:] - ballistic_final[3:]
    elif mode == "configured":
        burn1 = np.asarray(opt_cfg.get("first_burn_initial_guess_mps", [0, 0, 0]), dtype=float)
        burn2 = np.asarray(opt_cfg.get("second_burn_initial_guess_mps", [0, 0, 0]), dtype=float)
        if burn1.ndim == 0:
            burn1 = np.repeat(burn1, 3)
        if burn2.ndim == 0:
            burn2 = np.repeat(burn2, 3)
        if burn1.shape != (3,) or burn2.shape != (3,):
            raise ValueError("Configured burn initial guesses must be scalars or 3-vectors.")
        guess = np.hstack([burn1, burn2]) / 1000.0
    else:
        raise ValueError(f"Unknown optimization.initial_guess_mode: {mode}")

    epsilon = 1.0e-12
    guess = np.clip(guess, lower + epsilon, upper - epsilon)
    return guess, lower, upper


def solve_fixed_recovery_time(
    sample: dict[str, Any],
    transfer_days: float,
    state_commissioning_end: np.ndarray,
    t_burn1: float,
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    config: dict[str, Any],
    sample_deadline: float,
) -> dict[str, Any]:
    capture_jdtdb = float(t_burn1 + transfer_days)
    target_state = np.asarray(interpolate_reference_states(reference, capture_jdtdb), dtype=float)
    opt_cfg = config["optimization"]
    r_scale = float(opt_cfg["r_scale_km"])
    v_scale = float(opt_cfg["v_scale_kms"])
    evaluation_count = 0
    propagation_count = 0

    guess, lower, upper = optimizer_bounds_and_guess(
        config,
        state_commissioning_end,
        t_burn1,
        capture_jdtdb,
        target_state,
        propagator,
    )
    propagation_count += 1 if str(opt_cfg.get("initial_guess_mode", "")).lower() == "ballistic_informed" else 0

    def residual(decision: np.ndarray) -> np.ndarray:
        nonlocal evaluation_count, propagation_count
        if time.monotonic() > sample_deadline:
            raise SampleTimeoutError("Dispersion sample exceeded its runtime limit.")
        evaluation_count += 1
        state1_plus = np.asarray(state_commissioning_end, dtype=float).copy()
        state1_plus[3:] += np.asarray(decision[:3], dtype=float)
        state2_minus = propagate_final_state(
            propagator, state1_plus, t_burn1, capture_jdtdb
        )
        propagation_count += 1
        if time.monotonic() > sample_deadline:
            raise SampleTimeoutError("Dispersion sample exceeded its runtime limit.")
        state2_plus = state2_minus.copy()
        state2_plus[3:] += np.asarray(decision[3:], dtype=float)
        dr = state2_plus[:3] - target_state[:3]
        dv = state2_plus[3:] - target_state[3:]
        return np.hstack([dr / r_scale, dv / v_scale])

    start = time.monotonic()
    result = least_squares(
        residual,
        x0=guess,
        bounds=(lower, upper),
        method=str(opt_cfg.get("method", "trf")),
        max_nfev=int(opt_cfg.get("max_nfev", 300)),
        xtol=float(opt_cfg.get("xtol", 1.0e-10)),
        ftol=float(opt_cfg.get("ftol", 1.0e-10)),
        gtol=float(opt_cfg.get("gtol", 1.0e-10)),
        verbose=0,
    )
    runtime_seconds = time.monotonic() - start

    decision = np.asarray(result.x, dtype=float)
    state1_plus = np.asarray(state_commissioning_end, dtype=float).copy()
    state1_plus[3:] += decision[:3]
    state2_minus = propagate_final_state(
        propagator, state1_plus, t_burn1, capture_jdtdb
    )
    propagation_count += 1
    state2_plus = state2_minus.copy()
    state2_plus[3:] += decision[3:]
    dr = state2_plus[:3] - target_state[:3]
    dv = state2_plus[3:] - target_state[3:]

    burn1_mps = 1000.0 * decision[:3]
    burn2_mps = 1000.0 * decision[3:]
    return {
        "sample_id": int(sample["sample_id"]),
        "case_id": str(sample["case_id"]),
        "recovery_transfer_days": float(transfer_days),
        "capture_jdtdb": capture_jdtdb,
        "capture_utc": jdtdb_to_utc(capture_jdtdb),
        "optimizer_success": bool(result.success),
        "optimization_status": int(result.status),
        "optimization_message": str(result.message),
        "least_squares_cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
        "optimization_runtime_seconds": float(runtime_seconds),
        "objective_evaluations": int(evaluation_count),
        "post_burn1_propagations": int(propagation_count),
        "burn1_dv_x_kms": float(decision[0]),
        "burn1_dv_y_kms": float(decision[1]),
        "burn1_dv_z_kms": float(decision[2]),
        "burn1_dv_x_mps": float(burn1_mps[0]),
        "burn1_dv_y_mps": float(burn1_mps[1]),
        "burn1_dv_z_mps": float(burn1_mps[2]),
        "burn1_dv_mag_mps": float(np.linalg.norm(burn1_mps)),
        "burn2_dv_x_kms": float(decision[3]),
        "burn2_dv_y_kms": float(decision[4]),
        "burn2_dv_z_kms": float(decision[5]),
        "burn2_dv_x_mps": float(burn2_mps[0]),
        "burn2_dv_y_mps": float(burn2_mps[1]),
        "burn2_dv_z_mps": float(burn2_mps[2]),
        "burn2_dv_mag_mps": float(np.linalg.norm(burn2_mps)),
        "dv_total_mps": float(np.linalg.norm(burn1_mps) + np.linalg.norm(burn2_mps)),
        "final_position_error_x_km": float(dr[0]),
        "final_position_error_y_km": float(dr[1]),
        "final_position_error_z_km": float(dr[2]),
        "final_position_error_norm_km": float(np.linalg.norm(dr)),
        "final_velocity_error_x_mps": float(1000.0 * dv[0]),
        "final_velocity_error_y_mps": float(1000.0 * dv[1]),
        "final_velocity_error_z_mps": float(1000.0 * dv[2]),
        "final_velocity_error_norm_mps": float(1000.0 * np.linalg.norm(dv)),
        **{f"target_state_{i}": float(target_state[i]) for i in range(6)},
        **{f"final_state_{i}": float(state2_plus[i]) for i in range(6)},
    }


def sample_base_record(
    sample: dict[str, Any], reference: ReferenceOrbit, config: dict[str, Any]
) -> dict[str, Any]:
    commissioning = float(config["recovery"]["commissioning_duration_days"])
    t_burn1 = reference.deployment_jdtdb + commissioning
    return {
        **sample,
        "deployment_index": int(reference.deployment_index),
        "deployment_jdtdb": float(reference.deployment_jdtdb),
        "deployment_utc": reference.deployment_utc,
        "commissioning_duration_days": commissioning,
        "first_burn_jdtdb": float(t_burn1),
        "first_burn_utc": jdtdb_to_utc(t_burn1),
        "status": "NOT_RUN",
        "error_message": "",
        "optimizer_success": False,
        "sample_timed_out": False,
        "sample_runtime_seconds": np.nan,
        "recovery_times_attempted": 0,
        "recovery_times_optimizer_success": 0,
    }


def run_one_dispersion_sample(
    sample: dict[str, Any],
    recovery_times_days: np.ndarray,
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = sample_base_record(sample, reference, config)
    trials: list[dict[str, Any]] = []
    start = time.monotonic()
    max_minutes = float(config["runtime"]["max_sample_minutes"])
    deadline = start + 60.0 * max_minutes

    try:
        elevation = float(sample["separation_elevation_deg"])
        speed = float(sample["separation_speed_mps"])
        if not (-90.0 <= elevation <= 90.0):
            raise ValueError("Dispersed elevation lies outside [-90, 90] degrees.")
        if speed <= float(config["dispersion"].get("minimum_allowed_speed_mps", 0.0)):
            raise ValueError("Dispersed separation speed is not physically valid.")

        state0, separation_secr, separation_eme = build_post_separation_state(
            sample, reference
        )
        commissioning = float(config["recovery"]["commissioning_duration_days"])
        t_burn1 = reference.deployment_jdtdb + commissioning
        state_commissioning_end = propagate_final_state(
            propagator, state0, reference.deployment_jdtdb, t_burn1
        )
        if time.monotonic() > deadline:
            raise SampleTimeoutError("Dispersion sample exceeded its runtime limit.")

        record.update(
            {
                **{f"initial_state_{i}": float(state0[i]) for i in range(6)},
                **{
                    f"commissioning_end_state_{i}": float(state_commissioning_end[i])
                    for i in range(6)
                },
                "separation_dv_secr_x_mps": float(1000.0 * separation_secr[0]),
                "separation_dv_secr_y_mps": float(1000.0 * separation_secr[1]),
                "separation_dv_secr_z_mps": float(1000.0 * separation_secr[2]),
                "separation_dv_eme_x_mps": float(1000.0 * separation_eme[0]),
                "separation_dv_eme_y_mps": float(1000.0 * separation_eme[1]),
                "separation_dv_eme_z_mps": float(1000.0 * separation_eme[2]),
            }
        )

        for transfer_days in recovery_times_days:
            if time.monotonic() > deadline:
                raise SampleTimeoutError("Dispersion sample exceeded its runtime limit.")
            record["recovery_times_attempted"] += 1
            try:
                trial = solve_fixed_recovery_time(
                    sample=sample,
                    transfer_days=float(transfer_days),
                    state_commissioning_end=state_commissioning_end,
                    t_burn1=t_burn1,
                    reference=reference,
                    propagator=propagator,
                    config=config,
                    sample_deadline=deadline,
                )
                trial["trial_status"] = "OK" if trial["optimizer_success"] else "OPTIMIZER_FAILED"
                trial["trial_error_message"] = ""
                trials.append(trial)
                if trial["optimizer_success"]:
                    record["recovery_times_optimizer_success"] += 1
            except SampleTimeoutError:
                raise
            except Exception as exc:  # retain failed transfer-time trials
                trials.append(
                    {
                        "sample_id": int(sample["sample_id"]),
                        "case_id": str(sample["case_id"]),
                        "recovery_transfer_days": float(transfer_days),
                        "capture_jdtdb": float(t_burn1 + transfer_days),
                        "capture_utc": jdtdb_to_utc(t_burn1 + transfer_days),
                        "optimizer_success": False,
                        "trial_status": "ERROR",
                        "trial_error_message": f"{type(exc).__name__}: {exc}",
                    }
                )

        successful = [
            trial
            for trial in trials
            if bool(trial.get("optimizer_success", False))
            and np.isfinite(float(trial.get("dv_total_mps", np.nan)))
        ]
        if successful:
            successful.sort(
                key=lambda item: (
                    float(item["dv_total_mps"]),
                    float(item["least_squares_cost"]),
                    float(item["recovery_transfer_days"]),
                )
            )
            best = successful[0]
            record.update(best)
            record["status"] = "OK"
            record["error_message"] = ""
        else:
            record["status"] = "NO_SUCCESSFUL_RECOVERY"
            record["error_message"] = "No recovery duration returned optimizer_success=True."

    except SampleTimeoutError as exc:
        record["status"] = "PARTIAL_TIMEOUT" if trials else "TIMEOUT"
        record["sample_timed_out"] = True
        record["error_message"] = str(exc)
        successful = [
            trial
            for trial in trials
            if bool(trial.get("optimizer_success", False))
            and np.isfinite(float(trial.get("dv_total_mps", np.nan)))
        ]
        if successful:
            successful.sort(key=lambda item: float(item["dv_total_mps"]))
            record.update(successful[0])
            record["status"] = "PARTIAL_TIMEOUT_WITH_SOLUTION"
            record["sample_timed_out"] = True
    except Exception as exc:
        record["status"] = "ERROR"
        record["error_message"] = f"{type(exc).__name__}: {exc}"

    record["sample_runtime_seconds"] = float(time.monotonic() - start)
    return record, trials


# =============================================================================
# MPI/output helpers
# =============================================================================


def inside_walltime_buffer(run_start: float, config: dict[str, Any]) -> bool:
    runtime_cfg = config["runtime"]
    walltime_seconds = 3600.0 * float(runtime_cfg["walltime_hours"])
    buffer_seconds = 60.0 * float(runtime_cfg["save_before_walltime_minutes"])
    return (time.monotonic() - run_start) >= (walltime_seconds - buffer_seconds)


def save_rank_outputs(
    output_dir: Path,
    rank: int,
    summary_records: list[dict[str, Any]],
    trial_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    if not as_bool(config["mpi"].get("save_rank_outputs", True)):
        return
    rank_dir = output_dir / str(config["mpi"].get("rank_output_subdir", "rank_outputs"))
    rank_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_records).to_csv(
        rank_dir / f"recovery_dispersion_rank_{rank:04d}_summary.csv", index=False
    )
    if as_bool(config["output"].get("save_all_recovery_trials", True)):
        pd.DataFrame(trial_records).to_csv(
            rank_dir / f"recovery_dispersion_rank_{rank:04d}_trials.csv", index=False
        )


def assign_summary_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["dv_total_rank"] = np.nan
    summary["dv_total_percentile"] = np.nan
    valid = summary[
        summary["optimizer_success"].fillna(False)
        & summary["dv_total_mps"].notna()
    ].copy()
    if valid.empty:
        return summary
    order = valid.sort_values(["dv_total_mps", "sample_id"]).index
    count = len(order)
    ranks = np.arange(1, count + 1, dtype=float)
    percentiles = np.zeros(count) if count == 1 else 100.0 * (ranks - 1.0) / (count - 1.0)
    summary.loc[order, "dv_total_rank"] = ranks
    summary.loc[order, "dv_total_percentile"] = percentiles
    return summary


def write_resolved_config(config: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "resolved_config.yaml"
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return path


# =============================================================================
# Representative trajectories and plots
# =============================================================================


def select_representative_rows(summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    valid = summary[
        summary["optimizer_success"].fillna(False)
        & summary["dv_total_mps"].notna()
    ].copy()
    if valid.empty or not as_bool(config["output"].get("save_representative_trajectories", True)):
        return valid.iloc[0:0]
    percentiles = config["output"].get("representative_dv_percentiles", [0, 50, 95, 100])
    values = valid["dv_total_mps"].to_numpy(dtype=float)
    selected = []
    for percentile in percentiles:
        target = float(np.percentile(values, float(percentile)))
        distances = np.abs(values - target)
        order = np.lexsort((valid["sample_id"].to_numpy(dtype=int), distances))
        row = valid.iloc[int(order[0])].copy()
        row["selected_percentile"] = float(percentile)
        selected.append(row)
    return pd.DataFrame(selected).drop_duplicates(subset=["sample_id"]).reset_index(drop=True)


def trajectory_products_for_row(
    row: pd.Series,
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = row.to_dict()
    state0, _, _ = build_post_separation_state(sample, reference)
    t0 = reference.deployment_jdtdb
    t1 = float(row["first_burn_jdtdb"])
    tc = float(row["capture_jdtdb"])
    step = float(config["output"].get("trajectory_step_days", 0.05))
    times = np.unique(np.append(build_time_grid(t0, tc, step), [t1, tc]))
    times.sort()

    burn1 = np.asarray(
        [row["burn1_dv_x_kms"], row["burn1_dv_y_kms"], row["burn1_dv_z_kms"]],
        dtype=float,
    )
    burn2 = np.asarray(
        [row["burn2_dv_x_kms"], row["burn2_dv_y_kms"], row["burn2_dv_z_kms"]],
        dtype=float,
    )

    ballistic = np.asarray(
        propagator.propagate(x0_km=state0, t0_jdtdb=t0, t1_jdtdb=times),
        dtype=float,
    )
    optimized = np.empty((len(times), 6), dtype=float)
    pre_mask = times <= t1 + 1.0e-13
    post_mask = ~pre_mask
    optimized[pre_mask] = np.asarray(
        propagator.propagate(x0_km=state0, t0_jdtdb=t0, t1_jdtdb=times[pre_mask]),
        dtype=float,
    )
    state1_minus = propagate_final_state(propagator, state0, t0, t1)
    state1_plus = state1_minus.copy()
    state1_plus[3:] += burn1
    if np.any(post_mask):
        optimized[post_mask] = np.asarray(
            propagator.propagate(
                x0_km=state1_plus, t0_jdtdb=t1, t1_jdtdb=times[post_mask]
            ),
            dtype=float,
        )
    burn1_index = int(np.argmin(np.abs(times - t1)))
    optimized[burn1_index] = state1_plus
    state2_minus = propagate_final_state(propagator, state1_plus, t1, tc)
    state2_plus = state2_minus.copy()
    state2_plus[3:] += burn2
    optimized[-1] = state2_plus

    reference_states = np.asarray(interpolate_reference_states(reference, times), dtype=float)
    integrated_reference = np.asarray(
        propagator.propagate(
            x0_km=reference.deployment_state_eme,
            t0_jdtdb=t0,
            t1_jdtdb=times,
        ),
        dtype=float,
    )

    trajectory = pd.DataFrame(
        {
            "jdtdb": times,
            "utc": [jdtdb_to_utc(value) for value in times],
            "t_days_since_deployment": times - t0,
            **{f"ballistic_state_{i}": ballistic[:, i] for i in range(6)},
            **{f"optimized_state_{i}": optimized[:, i] for i in range(6)},
            **{f"lpf_reference_state_{i}": reference_states[:, i] for i in range(6)},
            **{f"integrated_reference_state_{i}": integrated_reference[:, i] for i in range(6)},
            "is_first_burn_epoch": np.isclose(times, t1, atol=1.0e-12, rtol=0.0),
            "is_capture_epoch": np.isclose(times, tc, atol=1.0e-12, rtol=0.0),
        }
    )
    trajectory["microsat_to_lpf_position_error_km"] = np.linalg.norm(
        optimized[:, :3] - reference_states[:, :3], axis=1
    )
    trajectory["integrated_minus_lpf_position_error_km"] = np.linalg.norm(
        integrated_reference[:, :3] - reference_states[:, :3], axis=1
    )

    maneuvers = pd.DataFrame(
        [
            {
                "name": "BURN1",
                "jdtdb": t1,
                "utc": jdtdb_to_utc(t1),
                "dv_x_kms": burn1[0],
                "dv_y_kms": burn1[1],
                "dv_z_kms": burn1[2],
                "dv_mag_mps": 1000.0 * np.linalg.norm(burn1),
                **{f"state_minus_{i}": state1_minus[i] for i in range(6)},
                **{f"state_plus_{i}": state1_plus[i] for i in range(6)},
            },
            {
                "name": "CAPTURE",
                "jdtdb": tc,
                "utc": jdtdb_to_utc(tc),
                "dv_x_kms": burn2[0],
                "dv_y_kms": burn2[1],
                "dv_z_kms": burn2[2],
                "dv_mag_mps": 1000.0 * np.linalg.norm(burn2),
                **{f"state_minus_{i}": state2_minus[i] for i in range(6)},
                **{f"state_plus_{i}": state2_plus[i] for i in range(6)},
            },
        ]
    )
    return trajectory, maneuvers


def plot_trajectory(
    trajectory: pd.DataFrame,
    row: pd.Series,
    path: Path,
    config: dict[str, Any],
) -> None:
    plot_cfg = config["plots"]
    width = float(plot_cfg["figure_size_inches"]["width"])
    height = float(plot_cfg["figure_size_inches"]["height"])
    plt.rcParams.update({"font.size": float(plot_cfg["font_size"])})
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        trajectory["ballistic_state_0"], trajectory["ballistic_state_1"], trajectory["ballistic_state_2"],
        linestyle=":", label="Ballistic microsatellite"
    )
    ax.plot(
        trajectory["optimized_state_0"], trajectory["optimized_state_1"], trajectory["optimized_state_2"],
        label="Recovered microsatellite"
    )
    ax.plot(
        trajectory["lpf_reference_state_0"], trajectory["lpf_reference_state_1"], trajectory["lpf_reference_state_2"],
        label="LPF reference"
    )
    ax.plot(
        trajectory["integrated_reference_state_0"], trajectory["integrated_reference_state_1"], trajectory["integrated_reference_state_2"],
        linestyle="--", label="Integrated reference"
    )
    burn_row = trajectory.loc[trajectory["is_first_burn_epoch"]].iloc[0]
    final = trajectory.iloc[-1]
    ax.scatter([0], [0], [0], s=40, label="Earth")
    ax.scatter(
        [burn_row["optimized_state_0"]], [burn_row["optimized_state_1"]], [burn_row["optimized_state_2"]],
        marker="^", s=55, label="Burn 1"
    )
    ax.scatter(
        [final["lpf_reference_state_0"]], [final["lpf_reference_state_1"]], [final["lpf_reference_state_2"]],
        marker="*", s=85, label="Capture target"
    )
    ax.set_xlabel("GEO EME X [km]")
    ax.set_ylabel("GEO EME Y [km]")
    ax.set_zlabel("GEO EME Z [km]")
    ax.set_title(
        f"Recovery sample {int(row['sample_id'])}\n"
        f"T={float(row['recovery_transfer_days']):.2f} d, ΔV={float(row['dv_total_mps']):.3f} m/s"
    )
    set_axes_equal_3d(ax)
    ax.legend()
    fig.tight_layout()
    if as_bool(plot_cfg.get("save_representative_trajectories", True)):
        fig.savefig(path, format=str(plot_cfg.get("format", "svg")), bbox_inches="tight")
    if as_bool(plot_cfg.get("show_plots", False)):
        plt.show()
    plt.close(fig)


def simple_plot(
    x: np.ndarray,
    y: np.ndarray | None,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    config: dict[str, Any],
    kind: str,
) -> None:
    plot_cfg = config["plots"]
    width = float(plot_cfg["figure_size_inches"]["width"])
    height = float(plot_cfg["figure_size_inches"]["height"])
    plt.rcParams.update({"font.size": float(plot_cfg["font_size"])})
    fig, ax = plt.subplots(figsize=(width, height))
    if kind == "hist":
        ax.hist(x, bins=int(plot_cfg.get("histogram_bins", 30)))
    elif kind == "scatter":
        ax.scatter(x, y)
    else:
        raise ValueError(f"Unknown plot kind: {kind}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, format=str(plot_cfg.get("format", "svg")), bbox_inches="tight")
    if as_bool(plot_cfg.get("show_plots", False)):
        plt.show()
    plt.close(fig)


def save_plots_and_representatives(
    summary: pd.DataFrame,
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not as_bool(config["plots"].get("enabled", True)):
        return [], []
    plot_dir = output_dir / "plots"
    traj_dir = output_dir / "representative_trajectories"
    plot_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    extension = str(config["plots"].get("format", "svg"))
    saved_plots: list[str] = []
    products: list[dict[str, Any]] = []

    valid = summary[
        summary["optimizer_success"].fillna(False)
        & summary["dv_total_mps"].notna()
    ].copy()
    if not valid.empty:
        plot_specs = [
            ("dv_histogram", valid["dv_total_mps"].to_numpy(float), None, "Total recovery ΔV [m/s]", "Count", "Recovery ΔV distribution", "hist"),
            ("recovery_time_histogram", valid["recovery_transfer_days"].to_numpy(float), None, "Selected recovery transfer time [days]", "Count", "Selected recovery-time distribution", "hist"),
            ("dv_vs_azimuth_error", valid["azimuth_error_deg"].to_numpy(float), valid["dv_total_mps"].to_numpy(float), "Azimuth error [deg]", "Total recovery ΔV [m/s]", "Recovery ΔV versus azimuth error", "scatter"),
            ("dv_vs_elevation_error", valid["elevation_error_deg"].to_numpy(float), valid["dv_total_mps"].to_numpy(float), "Elevation error [deg]", "Total recovery ΔV [m/s]", "Recovery ΔV versus elevation error", "scatter"),
            ("dv_vs_speed_error", valid["separation_speed_error_mps"].to_numpy(float), valid["dv_total_mps"].to_numpy(float), "Separation-speed error [m/s]", "Total recovery ΔV [m/s]", "Recovery ΔV versus separation-speed error", "scatter"),
        ]
        for name, x, y, xlabel, ylabel, title, kind in plot_specs:
            path = plot_dir / f"{name}.{extension}"
            simple_plot(x, y, xlabel, ylabel, title, path, config, kind)
            saved_plots.append(str(path))

    for _, row in select_representative_rows(summary, config).iterrows():
        percentile = float(row["selected_percentile"])
        stem = f"recovery_p{percentile:05.1f}_sample_{int(row['sample_id']):06d}"
        trajectory, maneuvers = trajectory_products_for_row(
            row, reference, propagator, config
        )
        trajectory_path = traj_dir / f"{stem}_trajectory.csv"
        maneuver_path = traj_dir / f"{stem}_maneuvers.csv"
        plot_path = plot_dir / f"{stem}_trajectory.{extension}"
        trajectory.to_csv(trajectory_path, index=False)
        maneuvers.to_csv(maneuver_path, index=False)
        plot_trajectory(trajectory, row, plot_path, config)
        if as_bool(config["plots"].get("save_representative_trajectories", True)):
            saved_plots.append(str(plot_path))
        products.append(
            {
                "sample_id": int(row["sample_id"]),
                "selected_percentile": percentile,
                "trajectory_csv": str(trajectory_path),
                "maneuver_csv": str(maneuver_path),
                "plot": str(plot_path),
            }
        )
    return saved_plots, products


# =============================================================================
# Metadata and main
# =============================================================================


def write_metadata(
    config_path: Path,
    config: dict[str, Any],
    summary: pd.DataFrame,
    trials: pd.DataFrame,
    reference: ReferenceOrbit,
    recovery_times: np.ndarray,
    output_dir: Path,
    mpi_size: int,
    runtime_seconds: float,
    stopped_flags: list[bool],
    assigned_counts: list[int],
    completed_counts: list[int],
    plot_files: list[str],
    representative_products: list[dict[str, Any]],
) -> Path:
    valid = summary[
        summary["optimizer_success"].fillna(False)
        & summary["dv_total_mps"].notna()
    ].copy()
    dv_stats = None
    if not valid.empty:
        values = valid["dv_total_mps"].to_numpy(dtype=float)
        dv_stats = {
            "minimum_mps": float(np.min(values)),
            "median_mps": float(np.median(values)),
            "mean_mps": float(np.mean(values)),
            "p95_mps": float(np.percentile(values, 95)),
            "maximum_mps": float(np.max(values)),
        }
    metadata = {
        "script": Path(__file__).name,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "reference_csv": str(config["files"]["lpf_orbit_csv"]),
        "reference_csv_sha256": sha256_file(config["files"]["lpf_orbit_csv"]),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "mpi_ranks": int(mpi_size),
        "runtime_seconds": float(runtime_seconds),
        "partial_run": bool(any(stopped_flags)),
        "rank_stopped_for_walltime": [bool(item) for item in stopped_flags],
        "rank_assigned_counts": [int(item) for item in assigned_counts],
        "rank_completed_counts": [int(item) for item in completed_counts],
        "n_samples_requested": int(sum(assigned_counts)),
        "n_samples_completed": int(len(summary)),
        "n_trial_rows": int(len(trials)),
        "status_counts": {
            str(key): int(value)
            for key, value in summary["status"].value_counts(dropna=False).to_dict().items()
        },
        "optimizer_success_count": int(summary["optimizer_success"].fillna(False).sum()),
        "reference_period": {
            "start_index": int(reference.start_index),
            "end_index_inclusive": int(reference.end_index),
            "deployment_index": int(reference.deployment_index),
            "period_days": float(reference.period_days),
            "deployment_jdtdb": float(reference.deployment_jdtdb),
            "deployment_utc": reference.deployment_utc,
        },
        "recovery_transfer_time_grid_days": recovery_times.tolist(),
        "dv_statistics": dv_stats,
        "plot_files": plot_files,
        "representative_products": representative_products,
    }
    path = output_dir / "recovery_dispersion_metadata.json"
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(json_safe(metadata), stream, indent=2)
    return path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    run_start = time.monotonic()

    try:
        load_spice_kernels(config)
        reference = load_reference_orbit(config)
        recovery_times = build_recovery_time_grid(config, reference)
        propagator = build_nbody_propagator(config)

        if rank == 0:
            samples = generate_dispersion_samples(config)
            output_dir = Path(config["files"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            write_resolved_config(config, output_dir)
            rank_print(
                rank,
                f"Generated {len(samples)} samples and {len(recovery_times)} recovery times per sample.",
            )
        else:
            samples = None
            output_dir = Path(config["files"]["output_dir"])

        samples = comm.bcast(samples, root=0)
        local_samples = samples[rank::size]
        local_summary: list[dict[str, Any]] = []
        local_trials: list[dict[str, Any]] = []
        stopped_for_walltime = False
        progress_every = max(1, int(config["mpi"].get("progress_every", 10)))
        save_every = max(1, int(config["mpi"].get("save_every_samples", progress_every)))

        for local_index, sample in enumerate(local_samples, start=1):
            if inside_walltime_buffer(run_start, config):
                stopped_for_walltime = True
                rank_print(rank, "Walltime save buffer reached; stopping before next sample.")
                break
            record, trials = run_one_dispersion_sample(
                sample, recovery_times, reference, propagator, config
            )
            local_summary.append(record)
            local_trials.extend(trials)
            if local_index % save_every == 0:
                save_rank_outputs(output_dir, rank, local_summary, local_trials, config)
            if as_bool(config["mpi"].get("rank_progress", True)) and (
                local_index % progress_every == 0 or local_index == len(local_samples)
            ):
                rank_print(
                    rank,
                    f"Completed {local_index}/{len(local_samples)} local samples; last status={record['status']}.",
                )

        save_rank_outputs(output_dir, rank, local_summary, local_trials, config)

        gathered_summary = comm.gather(local_summary, root=0)
        gathered_trials = comm.gather(local_trials, root=0)
        stopped_flags = comm.gather(stopped_for_walltime, root=0)
        assigned_counts = comm.gather(len(local_samples), root=0)
        completed_counts = comm.gather(len(local_summary), root=0)

        if rank == 0:
            summary_records = [item for group in gathered_summary for item in group]
            trial_records = [item for group in gathered_trials for item in group]
            summary = pd.DataFrame(summary_records)
            trials = pd.DataFrame(trial_records)
            if not summary.empty:
                summary = assign_summary_ranks(summary)
                summary = summary.sort_values("sample_id").reset_index(drop=True)
            if not trials.empty:
                trials = trials.sort_values(
                    ["sample_id", "recovery_transfer_days"], na_position="last"
                ).reset_index(drop=True)

            summary_path = output_dir / "recovery_dispersion_summary.csv"
            trials_path = output_dir / "recovery_dispersion_all_trials.csv"
            summary.to_csv(summary_path, index=False)
            if as_bool(config["output"].get("save_all_recovery_trials", True)):
                trials.to_csv(trials_path, index=False)

            plot_files, representative_products = save_plots_and_representatives(
                summary, reference, propagator, config, output_dir
            )
            metadata_path = write_metadata(
                config_path=config_path,
                config=config,
                summary=summary,
                trials=trials,
                reference=reference,
                recovery_times=recovery_times,
                output_dir=output_dir,
                mpi_size=size,
                runtime_seconds=time.monotonic() - run_start,
                stopped_flags=stopped_flags,
                assigned_counts=assigned_counts,
                completed_counts=completed_counts,
                plot_files=plot_files,
                representative_products=representative_products,
            )
            rank_print(rank, f"Wrote summary: {summary_path}")
            if as_bool(config["output"].get("save_all_recovery_trials", True)):
                rank_print(rank, f"Wrote all trials: {trials_path}")
            rank_print(rank, f"Wrote metadata: {metadata_path}")

    finally:
        spice.kclear()


if __name__ == "__main__":
    main()
