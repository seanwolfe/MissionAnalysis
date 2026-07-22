"""
MPI optimization and dispersion analysis for microsatellite deployment from a
mothership on a Sun--Earth L1 quasi-halo reference orbit.

Run modes
---------
1. nominal_sweep
   For each requested mothership--microsatellite phase separation:
     * derive the fixed acquisition epoch from the reference period;
     * sweep the optional coast/wait time after commissioning;
     * optimize five variables at each wait time:
           separation azimuth, separation elevation, and first-burn dV (3);
     * derive the second/acquisition burn from the terminal velocity mismatch;
     * select the feasible solution with minimum microsatellite corrective dV;
     * save one nominal master row and trajectory bundle per phase separation.

2. dispersion
   Load exactly one nominal phase result, hold its commanded separation
   direction and nominal wait time fixed, generate Monte Carlo deployment cases,
   optimize only the first correction burn for each case, derive the acquisition
   burn, assign empirical dV percentiles, and save only configured representative
   trajectories.

State convention
----------------
Earth-centred EME/J2000:
    [x, y, z, vx, vy, vz]
position [km], velocity [km/s].

External project modules expected beside this script (or on PYTHONPATH)
-----------------------------------------------------------------------
    n_body_integrator.py   -> NBodyPropagator
    utilities.py           -> frame-conversion functions supplied by the user

The frame-conversion calls intentionally use:
    import utilities as util
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import mpi4py.rc

mpi4py.rc.threads = False
from mpi4py import MPI

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as spice
import yaml
from scipy.optimize import least_squares, minimize

import utilities as util
from n_body_integrator import NBodyPropagator


# =============================================================================
# Exceptions and small helpers
# =============================================================================


class OptimizationTimeoutError(RuntimeError):
    """Raised when one optimization case exceeds its configured wallclock limit."""


class TemporalInfeasibilityError(RuntimeError):
    """Raised when a burn/commissioning epoch is not before acquisition."""


def unit(vector: np.ndarray, eps: float = 1.0e-15) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        raise ValueError("Cannot normalize a zero or near-zero vector.")
    return vector / norm


def smooth_norm(vector: np.ndarray, epsilon: float) -> float:
    vector = np.asarray(vector, dtype=float)
    return float(np.sqrt(np.dot(vector, vector) + float(epsilon) ** 2))


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    return bool(value)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a YAML mapping.")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, default=json_default)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename(text: str) -> str:
    cleaned = []
    for char in str(text):
        cleaned.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(cleaned)


def format_elapsed(seconds: float) -> str:
    seconds_i = int(max(0.0, seconds))
    hours = seconds_i // 3600
    minutes = (seconds_i % 3600) // 60
    seconds_remainder = seconds_i % 60
    return f"{hours:02d}:{minutes:02d}:{seconds_remainder:02d}"


def check_deadline(deadline: Optional[float]) -> None:
    if deadline is not None and time.time() >= deadline:
        raise OptimizationTimeoutError(
            "Optimization case exceeded runtime.max_optimization_minutes."
        )


def make_case_deadline(config: dict[str, Any], start_time: float) -> Optional[float]:
    max_minutes = config.get("runtime", {}).get("max_optimization_minutes", None)
    if max_minutes is None:
        return None
    max_minutes = float(max_minutes)
    if max_minutes <= 0.0:
        return None
    return float(start_time) + 60.0 * max_minutes


def should_save_and_stop(config: dict[str, Any], run_start_time: float) -> bool:
    runtime_cfg = config.get("runtime", {})
    walltime_hours = runtime_cfg.get("walltime_hours", None)
    if walltime_hours is None:
        return False
    walltime_seconds = 3600.0 * float(walltime_hours)
    buffer_seconds = 60.0 * float(
        runtime_cfg.get("save_before_walltime_minutes", 10.0)
    )
    elapsed = time.time() - float(run_start_time)
    return elapsed >= walltime_seconds - buffer_seconds


def load_spice_kernels(config: dict[str, Any]) -> None:
    for kernel in config["spice"]["kernels"]:
        spice.furnsh(str(kernel))


def utc_to_jdtdb(value: Any) -> float:
    timestamp = pd.to_datetime(str(value), utc=True)
    utc_clean = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")
    et = spice.str2et(utc_clean)
    return float(spice.unitim(et, "ET", "JDTDB"))


def jdtdb_to_et(jdtdb: float) -> float:
    return float(spice.unitim(float(jdtdb), "JDTDB", "ET"))


def jdtdb_to_utc(jdtdb: float) -> str:
    return spice.et2utc(jdtdb_to_et(float(jdtdb)), "ISOC", 6)


def get_earth_heliocentric_eclip_state(jdtdb: float) -> np.ndarray:
    """Earth state relative to the Sun in ECLIPJ2000 [km, km/s]."""
    state, _ = spice.spkgeo(399, jdtdb_to_et(jdtdb), "ECLIPJ2000", 10)
    return np.asarray(state, dtype=float)


def build_time_grid(t0: float, tf: float, step_days: float) -> np.ndarray:
    t0 = float(t0)
    tf = float(tf)
    step_days = float(step_days)
    if step_days <= 0.0:
        raise ValueError("Trajectory step_days must be positive.")
    if tf < t0:
        raise ValueError("Final trajectory epoch must not precede initial epoch.")
    if np.isclose(tf, t0):
        return np.asarray([t0], dtype=float)
    grid = np.arange(t0, tf, step_days, dtype=float)
    if grid.size == 0 or not np.isclose(grid[0], t0):
        grid = np.insert(grid, 0, t0)
    if not np.isclose(grid[-1], tf):
        grid = np.append(grid, tf)
    else:
        grid[-1] = tf
    return grid


def unique_sorted_times(*arrays: Iterable[float]) -> np.ndarray:
    values: list[float] = []
    for array in arrays:
        values.extend(float(item) for item in array)
    return np.asarray(sorted(set(values)), dtype=float)


# =============================================================================
# Reference orbit and frame handling
# =============================================================================


@dataclass
class ReferenceOrbit:
    dataframe: pd.DataFrame
    period_dataframe: pd.DataFrame
    time_column: str
    state_columns: list[str]
    start_index: int
    end_index: int
    start_jdtdb: float
    end_jdtdb: float
    period_days: float
    deployment_state_eme: np.ndarray
    phase_zero_state_secr: np.ndarray
    deployment_local_basis_secr: np.ndarray
    period_timestamps_utc: pd.DatetimeIndex
    period_elapsed_days_utc: np.ndarray


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
    """Rotate a pure SECR velocity increment to EME using a zero-position state."""
    dummy_state = np.hstack([np.zeros(3), np.asarray(vector_secr, dtype=float)])
    return state_secr_to_eme(dummy_state, jdtdb)[3:]


def vector_eme_to_secr(vector_eme: np.ndarray, jdtdb: float) -> np.ndarray:
    dummy_state = np.hstack([np.zeros(3), np.asarray(vector_eme, dtype=float)])
    return state_eme_to_secr(dummy_state, jdtdb)[3:]


def build_local_quasi_halo_basis(state_secr: np.ndarray) -> np.ndarray:
    """
    Return columns [tangent, cross, normal] in SECR.

    tangent = rotating-frame velocity direction
    normal  = normalized r x v
    cross   = normal x tangent
    """
    state_secr = np.asarray(state_secr, dtype=float)
    r = state_secr[:3]
    v = state_secr[3:]
    e_t = unit(v)
    e_n = unit(np.cross(r, v))
    e_c = unit(np.cross(e_n, e_t))
    basis = np.column_stack([e_t, e_c, e_n])
    if np.linalg.det(basis) < 0.0:
        e_c = -e_c
        basis = np.column_stack([e_t, e_c, e_n])
    return basis


def local_spherical_vector(
    speed_mps: float,
    azimuth_rad: float,
    elevation_rad: float,
    basis_secr: np.ndarray,
) -> np.ndarray:
    components = np.array(
        [
            math.cos(elevation_rad) * math.cos(azimuth_rad),
            math.cos(elevation_rad) * math.sin(azimuth_rad),
            math.sin(elevation_rad),
        ],
        dtype=float,
    )
    return (float(speed_mps) / 1000.0) * (basis_secr @ components)


def load_reference_orbit(config: dict[str, Any]) -> ReferenceOrbit:
    reference_cfg = config["reference_orbit"]
    path = Path(config["files"]["lpf_orbit_csv"])
    dataframe = pd.read_csv(path)

    time_column = reference_cfg["time_column"]
    state_columns = list(reference_cfg["state_columns_geo_eme"])
    required_columns = [time_column] + state_columns
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise KeyError(f"LPF reference CSV is missing columns: {missing}")

    start_index = int(reference_cfg["quasi_halo_start"])
    end_index = int(reference_cfg["quasi_halo_one_period_end"])
    if start_index < 0 or end_index >= len(dataframe) or end_index <= start_index:
        raise IndexError(
            "Invalid quasi-halo indices: "
            f"start={start_index}, end={end_index}, rows={len(dataframe)}"
        )

    # End index is explicitly inclusive.
    period_dataframe = dataframe.iloc[start_index : end_index + 1].copy()
    start_jdtdb = utc_to_jdtdb(period_dataframe.iloc[0][time_column])
    end_jdtdb = utc_to_jdtdb(period_dataframe.iloc[-1][time_column])
    period_days = end_jdtdb - start_jdtdb
    if period_days <= 0.0:
        raise ValueError("Reference quasi-halo period duration is not positive.")

    period_timestamps_utc = pd.DatetimeIndex(
        pd.to_datetime(
            period_dataframe[time_column].astype(str),
            utc=True,
        )
    )

    period_elapsed_days_utc = (
        period_timestamps_utc.astype('int64') - period_timestamps_utc[0].value
    ) / (86400.0 * 1.0e9)

    deployment_state = period_dataframe.iloc[0][state_columns].to_numpy(dtype=float)
    phase_zero_state_secr = state_eme_to_secr(deployment_state, start_jdtdb)
    local_basis = build_local_quasi_halo_basis(phase_zero_state_secr)

    return ReferenceOrbit(
        dataframe=dataframe,
        period_dataframe=period_dataframe,
        time_column=time_column,
        state_columns=state_columns,
        start_index=start_index,
        end_index=end_index,
        start_jdtdb=start_jdtdb,
        end_jdtdb=end_jdtdb,
        period_days=period_days,
        deployment_state_eme=deployment_state,
        phase_zero_state_secr=phase_zero_state_secr,
        deployment_local_basis_secr=local_basis,
        period_timestamps_utc=period_timestamps_utc,
        period_elapsed_days_utc=np.asarray(period_elapsed_days_utc, dtype=float),
    )


def interpolate_reference_eme_at_epoch(
    reference: ReferenceOrbit, jdtdb: float
) -> np.ndarray:
    """Linear UTC-time interpolation of the source LPF EME state for validation."""
    timestamp = pd.to_datetime(jdtdb_to_utc(jdtdb), utc=True)
    elapsed_days = (timestamp.value - reference.period_timestamps_utc[0].value) / (
        86400.0 * 1.0e9
    )
    times = reference.period_elapsed_days_utc
    if elapsed_days < times[0] or elapsed_days > times[-1]:
        raise ValueError("Requested validation epoch lies outside the reference period.")
    result = np.empty(6, dtype=float)
    for index, column in enumerate(reference.state_columns):
        values = reference.period_dataframe[column].to_numpy(dtype=float)
        result[index] = np.interp(elapsed_days, times, values)
    return result


def run_frame_validation(
    reference: ReferenceOrbit,
    acquisition_jdtdb: float,
) -> dict[str, float]:
    """
    Validate the supplied two-stage transformations at phase zero and at the
    acquisition epoch (equivalent to zero relative phase offset).
    """
    phase_zero_roundtrip = state_secr_to_eme(
        reference.phase_zero_state_secr, reference.start_jdtdb
    )
    phase_zero_dr = phase_zero_roundtrip[:3] - reference.deployment_state_eme[:3]
    phase_zero_dv = phase_zero_roundtrip[3:] - reference.deployment_state_eme[3:]

    direct_acquisition_state = interpolate_reference_eme_at_epoch(
        reference, acquisition_jdtdb
    )
    acquisition_secr = state_eme_to_secr(
        direct_acquisition_state, acquisition_jdtdb
    )
    acquisition_roundtrip = state_secr_to_eme(acquisition_secr, acquisition_jdtdb)
    acquisition_dr = acquisition_roundtrip[:3] - direct_acquisition_state[:3]
    acquisition_dv = acquisition_roundtrip[3:] - direct_acquisition_state[3:]

    return {
        "phase_zero_roundtrip_position_error_km": float(np.linalg.norm(phase_zero_dr)),
        "phase_zero_roundtrip_velocity_error_mps": float(
            1000.0 * np.linalg.norm(phase_zero_dv)
        ),
        "zero_relative_phase_position_error_km": float(
            np.linalg.norm(acquisition_dr)
        ),
        "zero_relative_phase_velocity_error_mps": float(
            1000.0 * np.linalg.norm(acquisition_dv)
        ),
    }


# =============================================================================
# Propagator
# =============================================================================


def load_nbody_constants(config: dict[str, Any]) -> dict[str, Any]:
    with open(config["files"]["constants_yaml"], "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_nbody_propagator(config: dict[str, Any]) -> NBodyPropagator:
    constants = load_nbody_constants(config)
    propagation_cfg = config["propagation"]
    return NBodyPropagator(
        spice=spice,
        config=constants,
        bodies=tuple(propagation_cfg["bodies"]),
        frame=propagation_cfg.get("frame", "J2000"),
        origin=propagation_cfg.get("origin", "399"),
        rtol=float(propagation_cfg.get("rtol", 1.0e-10)),
        atol=float(propagation_cfg.get("atol", 1.0e-12)),
        method=propagation_cfg.get("method", "DOP853"),
    )


def propagate_state(
    propagator: NBodyPropagator,
    state: np.ndarray,
    t0_jdtdb: float,
    t1_jdtdb: float,
) -> np.ndarray:
    """Propagate one state, safely handling a zero-duration interval."""
    if np.isclose(float(t0_jdtdb), float(t1_jdtdb), rtol=0.0, atol=1.0e-14):
        return np.asarray(state, dtype=float).copy()
    return np.asarray(
        propagator.propagate(
            x0_km=np.asarray(state, dtype=float),
            t0_jdtdb=float(t0_jdtdb),
            t1_jdtdb=float(t1_jdtdb),
        ),
        dtype=float,
    )


def propagate_grid(
    propagator: NBodyPropagator,
    state: np.ndarray,
    t0_jdtdb: float,
    times_jdtdb: np.ndarray,
) -> np.ndarray:
    """Propagate to a time grid, safely handling a one-point start grid."""
    times = np.asarray(times_jdtdb, dtype=float)
    if times.ndim != 1:
        raise ValueError("Propagation time grid must be one-dimensional.")
    if len(times) == 1 and np.isclose(
        float(t0_jdtdb), float(times[0]), rtol=0.0, atol=1.0e-14
    ):
        return np.asarray(state, dtype=float).reshape(1, 6).copy()
    return np.asarray(
        propagator.propagate(
            x0_km=np.asarray(state, dtype=float),
            t0_jdtdb=float(t0_jdtdb),
            t1_jdtdb=times,
        ),
        dtype=float,
    )


# =============================================================================
# Phase sweep and target timing
# =============================================================================


def requested_phase_values_rad(config: dict[str, Any]) -> list[float]:
    sweep_cfg = config["phase_sweep"]
    values_cfg = sweep_cfg["desired_separations"]
    mode = str(values_cfg.get("mode", "explicit_values")).lower()
    if mode == "explicit_values":
        values = np.asarray(values_cfg["values"], dtype=float)
    elif mode == "linspace":
        values = np.linspace(
            float(values_cfg["start"]),
            float(values_cfg["stop"]),
            int(values_cfg["count"]),
            endpoint=True,
        )
    else:
        raise ValueError(f"Unknown desired_separations.mode: {mode}")

    unit_name = str(values_cfg.get("unit", "rad")).lower()
    if unit_name in {"rad", "radian", "radians"}:
        values_rad = values
    elif unit_name in {"deg", "degree", "degrees"}:
        values_rad = np.deg2rad(values)
    elif unit_name in {"fraction", "cycle_fraction", "period_fraction"}:
        values_rad = 2.0 * np.pi * values
    else:
        raise ValueError(f"Unknown phase separation unit: {unit_name}")

    if np.any(values_rad <= 0.0):
        raise ValueError("Every desired phase separation must be positive.")
    if as_bool(sweep_cfg.get("require_within_one_period", True)) and np.any(
        values_rad > 2.0 * np.pi + 1.0e-12
    ):
        raise ValueError("Desired phase separation exceeds one reference period.")

    return [float(value) for value in values_rad]


def phase_context(
    phase_id: str,
    phase_rad: float,
    reference: ReferenceOrbit,
    config: dict[str, Any],
) -> dict[str, Any]:
    commissioning_days = float(config["commissioning"]["nominal_duration_days"])
    total_phase_days = float(phase_rad / (2.0 * np.pi) * reference.period_days)
    available_after_commissioning = total_phase_days - commissioning_days
    if available_after_commissioning <= 0.0:
        raise TemporalInfeasibilityError(
            f"Phase {phase_id}: commissioning duration ({commissioning_days:.6f} d) "
            f"is not shorter than phase duration ({total_phase_days:.6f} d)."
        )

    acquisition_jdtdb = reference.start_jdtdb + total_phase_days
    if acquisition_jdtdb > reference.end_jdtdb + 1.0e-12:
        raise TemporalInfeasibilityError(
            f"Phase {phase_id}: acquisition lies beyond the configured reference period."
        )

    target_state_eme = state_secr_to_eme(
        reference.phase_zero_state_secr, acquisition_jdtdb
    )
    validation = {}
    if as_bool(config.get("target_generation", {}).get("validate_frame_conversion", True)):
        validation = run_frame_validation(reference, acquisition_jdtdb)

    return {
        "separation_id": phase_id,
        "separation_phase_rad": float(phase_rad),
        "separation_phase_deg": float(np.rad2deg(phase_rad)),
        "separation_phase_fraction": float(phase_rad / (2.0 * np.pi)),
        "reference_period_days": float(reference.period_days),
        "deployment_jdtdb": float(reference.start_jdtdb),
        "deployment_utc": jdtdb_to_utc(reference.start_jdtdb),
        "commissioning_duration_days": commissioning_days,
        "total_deployment_to_acquisition_days": total_phase_days,
        "available_after_commissioning_days": available_after_commissioning,
        "acquisition_jdtdb": float(acquisition_jdtdb),
        "acquisition_utc": jdtdb_to_utc(acquisition_jdtdb),
        "target_state_eme": target_state_eme,
        **validation,
    }


def build_phase_contexts(
    reference: ReferenceOrbit, config: dict[str, Any]
) -> list[dict[str, Any]]:
    phases = requested_phase_values_rad(config)
    contexts: list[dict[str, Any]] = []
    for index, phase_rad in enumerate(phases):
        phase_id = f"sep_{index:03d}_{np.rad2deg(phase_rad):09.4f}deg"
        contexts.append(phase_context(phase_id, phase_rad, reference, config))
    return contexts


# =============================================================================
# Wait-time sweep generation
# =============================================================================


def coarse_wait_values(
    phase: dict[str, Any], config: dict[str, Any]
) -> np.ndarray:
    sweep_cfg = config["wait_time_sweep"]
    available = float(phase["available_after_commissioning_days"])
    minimum_post_burn = float(sweep_cfg["minimum_post_burn_coast_days"])
    physical_maximum = available - minimum_post_burn
    if physical_maximum < 0.0:
        raise TemporalInfeasibilityError(
            f"Phase {phase['separation_id']} does not leave the configured minimum "
            "post-burn coast after commissioning."
        )

    mode = str(sweep_cfg.get("bound_mode", "fraction_of_available_time")).lower()
    count = int(sweep_cfg["number_of_samples"])
    if count < 1:
        raise ValueError("wait_time_sweep.number_of_samples must be at least one.")

    if mode == "fraction_of_available_time":
        minimum_fraction = float(sweep_cfg.get("minimum_fraction", 0.0))
        maximum_fraction = float(sweep_cfg.get("maximum_fraction", 1.0))
        if not (0.0 <= minimum_fraction <= maximum_fraction <= 1.0):
            raise ValueError("Wait-time fractions must satisfy 0 <= min <= max <= 1.")
        values = np.linspace(
            minimum_fraction * physical_maximum,
            maximum_fraction * physical_maximum,
            count,
            endpoint=True,
        )
    elif mode == "absolute_days":
        minimum_days = float(sweep_cfg.get("minimum_wait_days", 0.0))
        maximum_days = min(
            float(sweep_cfg.get("maximum_wait_days", physical_maximum)),
            physical_maximum,
        )
        if maximum_days < minimum_days:
            raise TemporalInfeasibilityError(
                f"Phase {phase['separation_id']} effective maximum wait is below minimum wait."
            )
        values = np.linspace(minimum_days, maximum_days, count, endpoint=True)
    else:
        raise ValueError(f"Unknown wait_time_sweep.bound_mode: {mode}")

    return np.unique(np.round(values.astype(float), decimals=14))


def build_coarse_wait_tasks(
    phase_contexts: list[dict[str, Any]], config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(phase_contexts):
        wait_values = coarse_wait_values(phase, config)
        physical_range = max(float(wait_values[-1] - wait_values[0]), 0.0)
        for wait_index, wait_days in enumerate(wait_values):
            wait_fraction = (
                0.0
                if physical_range <= 0.0
                else float((wait_days - wait_values[0]) / physical_range)
            )
            rows.append(
                {
                    "task_id": f"{phase['separation_id']}_coarse_{wait_index:04d}",
                    "stage": "coarse",
                    "phase_index": phase_index,
                    "separation_id": phase["separation_id"],
                    "separation_phase_rad": phase["separation_phase_rad"],
                    "wait_index": wait_index,
                    "wait_after_commissioning_days": float(wait_days),
                    "wait_fraction_within_configured_sweep": wait_fraction,
                }
            )
    return pd.DataFrame(rows)


def successful_wait_subset(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    mask = (dataframe["optimizer_success"] == True) & (
        dataframe["target_success"] == True
    )
    return dataframe[mask].copy()


def build_refinement_wait_tasks(
    coarse_results: pd.DataFrame,
    phase_contexts: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    refinement_cfg = config["wait_time_sweep"].get("refinement", {})
    if not as_bool(refinement_cfg.get("enabled", True)):
        return pd.DataFrame()

    n_refine = int(refinement_cfg.get("number_of_samples", 11))
    neighbour_intervals = int(refinement_cfg.get("number_of_neighbour_intervals", 1))
    if n_refine < 2:
        return pd.DataFrame()

    phase_lookup = {phase["separation_id"]: phase for phase in phase_contexts}
    rows: list[dict[str, Any]] = []

    for separation_id, group in coarse_results.groupby("separation_id", sort=False):
        group = group.sort_values("wait_after_commissioning_days").reset_index(drop=True)
        successful = successful_wait_subset(group)
        if successful.empty:
            continue
        best_original_index = int(successful["dv_total_mps"].idxmin())
        best_wait = float(coarse_results.loc[best_original_index, "wait_after_commissioning_days"])
        local_best_index = int(
            np.argmin(
                np.abs(group["wait_after_commissioning_days"].to_numpy(dtype=float) - best_wait)
            )
        )
        lower_index = max(0, local_best_index - neighbour_intervals)
        upper_index = min(len(group) - 1, local_best_index + neighbour_intervals)
        lower = float(group.iloc[lower_index]["wait_after_commissioning_days"])
        upper = float(group.iloc[upper_index]["wait_after_commissioning_days"])
        if np.isclose(lower, upper):
            continue

        existing = group["wait_after_commissioning_days"].to_numpy(dtype=float)
        candidates = np.linspace(lower, upper, n_refine, endpoint=True)
        candidates = [
            float(value)
            for value in candidates
            if not np.any(np.isclose(existing, value, rtol=0.0, atol=1.0e-12))
        ]
        phase = phase_lookup[separation_id]
        for refine_index, wait_days in enumerate(candidates):
            rows.append(
                {
                    "task_id": f"{separation_id}_refine_{refine_index:04d}",
                    "stage": "refine",
                    "phase_index": int(
                        next(
                            index
                            for index, item in enumerate(phase_contexts)
                            if item["separation_id"] == separation_id
                        )
                    ),
                    "separation_id": separation_id,
                    "separation_phase_rad": phase["separation_phase_rad"],
                    "wait_index": refine_index,
                    "wait_after_commissioning_days": wait_days,
                    "wait_fraction_within_configured_sweep": np.nan,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Nominal five-variable optimization
# =============================================================================


class NominalFiveVariableEvaluator:
    """Cache propagation results shared by objective and constraints."""

    def __init__(
        self,
        propagator: NBodyPropagator,
        mothership_state_eme: np.ndarray,
        deployment_jdtdb: float,
        first_burn_jdtdb: float,
        acquisition_jdtdb: float,
        target_state_eme: np.ndarray,
        local_basis_secr: np.ndarray,
        fixed_separation_speed_mps: float,
        config: dict[str, Any],
        deadline: Optional[float],
    ) -> None:
        self.propagator = propagator
        self.mothership_state_eme = np.asarray(mothership_state_eme, dtype=float)
        self.deployment_jdtdb = float(deployment_jdtdb)
        self.first_burn_jdtdb = float(first_burn_jdtdb)
        self.acquisition_jdtdb = float(acquisition_jdtdb)
        self.target_state_eme = np.asarray(target_state_eme, dtype=float)
        self.local_basis_secr = np.asarray(local_basis_secr, dtype=float)
        self.fixed_separation_speed_mps = float(fixed_separation_speed_mps)
        self.config = config
        self.deadline = deadline
        self.last_x: Optional[np.ndarray] = None
        self.last_result: Optional[dict[str, Any]] = None

        # The separation-to-first-burn coast depends on the separation angles,
        # but not on dV1. Cache it so fixed-angle targeting does not repeat the
        # identical pre-burn propagation for every least-squares evaluation.
        self.preburn_cache_angles: Optional[np.ndarray] = None
        self.preburn_cache_separation_dv_secr: Optional[np.ndarray] = None
        self.preburn_cache_separation_dv_eme: Optional[np.ndarray] = None
        self.preburn_cache_state0: Optional[np.ndarray] = None
        self.preburn_cache_state_first_minus: Optional[np.ndarray] = None

        # Runtime diagnostics written to the output CSV.
        self.unique_trajectory_evaluations = 0
        self.preburn_propagations = 0
        self.postburn_propagations = 0

    def evaluate(self, x: np.ndarray) -> dict[str, Any]:
        check_deadline(self.deadline)
        x = np.asarray(x, dtype=float)
        if self.last_x is not None and np.array_equal(x, self.last_x):
            assert self.last_result is not None
            return self.last_result

        self.unique_trajectory_evaluations += 1

        alpha_rad = float(x[0])
        beta_rad = float(x[1])
        dv1_mps = np.asarray(x[2:5], dtype=float)
        angles = np.array([alpha_rad, beta_rad], dtype=float)

        if (
            self.preburn_cache_angles is not None
            and np.array_equal(angles, self.preburn_cache_angles)
        ):
            assert self.preburn_cache_separation_dv_secr is not None
            assert self.preburn_cache_separation_dv_eme is not None
            assert self.preburn_cache_state0 is not None
            assert self.preburn_cache_state_first_minus is not None
            separation_dv_secr = self.preburn_cache_separation_dv_secr
            separation_dv_eme = self.preburn_cache_separation_dv_eme
            state0 = self.preburn_cache_state0
            state_first_minus = self.preburn_cache_state_first_minus
        else:
            separation_dv_secr = local_spherical_vector(
                self.fixed_separation_speed_mps,
                alpha_rad,
                beta_rad,
                self.local_basis_secr,
            )
            separation_dv_eme = vector_secr_to_eme(
                separation_dv_secr, self.deployment_jdtdb
            )

            state0 = self.mothership_state_eme.copy()
            state0[3:] += separation_dv_eme
            state_first_minus = propagate_state(
                self.propagator,
                state0,
                self.deployment_jdtdb,
                self.first_burn_jdtdb,
            )
            self.preburn_propagations += 1
            self.preburn_cache_angles = angles.copy()
            self.preburn_cache_separation_dv_secr = np.asarray(
                separation_dv_secr, dtype=float
            ).copy()
            self.preburn_cache_separation_dv_eme = np.asarray(
                separation_dv_eme, dtype=float
            ).copy()
            self.preburn_cache_state0 = np.asarray(state0, dtype=float).copy()
            self.preburn_cache_state_first_minus = np.asarray(
                state_first_minus, dtype=float
            ).copy()

        state_first_plus = np.asarray(state_first_minus, dtype=float).copy()
        state_first_plus[3:] += dv1_mps / 1000.0

        state_acquisition_minus = propagate_state(
            self.propagator,
            state_first_plus,
            self.first_burn_jdtdb,
            self.acquisition_jdtdb,
        )
        self.postburn_propagations += 1
        dv2_kms = self.target_state_eme[3:] - state_acquisition_minus[3:]
        dv2_mps = 1000.0 * dv2_kms
        position_error = state_acquisition_minus[:3] - self.target_state_eme[:3]

        state_acquisition_plus = state_acquisition_minus.copy()
        state_acquisition_plus[3:] += dv2_kms
        velocity_error_mps = 1000.0 * (
            state_acquisition_plus[3:] - self.target_state_eme[3:]
        )

        result = {
            "alpha_rad": alpha_rad,
            "beta_rad": beta_rad,
            "dv1_mps": dv1_mps,
            "dv2_mps": dv2_mps,
            "dv1_mag_mps": float(np.linalg.norm(dv1_mps)),
            "dv2_mag_mps": float(np.linalg.norm(dv2_mps)),
            "dv_total_mps": float(np.linalg.norm(dv1_mps) + np.linalg.norm(dv2_mps)),
            "separation_dv_secr_kms": separation_dv_secr,
            "separation_dv_eme_kms": separation_dv_eme,
            "state0": state0,
            "state_first_minus": np.asarray(state_first_minus, dtype=float),
            "state_first_plus": state_first_plus,
            "state_acquisition_minus": state_acquisition_minus,
            "state_acquisition_plus": state_acquisition_plus,
            "position_error_km": position_error,
            "velocity_error_mps": velocity_error_mps,
        }
        self.last_x = x.copy()
        self.last_result = result
        check_deadline(self.deadline)
        return result

    def objective(self, x: np.ndarray) -> float:
        result = self.evaluate(x)
        epsilon = float(
            self.config["nominal_optimization"].get(
                "objective_velocity_smoothing_mps", 1.0e-9
            )
        )
        return smooth_norm(result["dv1_mps"], epsilon) + smooth_norm(
            result["dv2_mps"], epsilon
        )

    def position_constraint(self, x: np.ndarray) -> np.ndarray:
        result = self.evaluate(x)
        scale = float(
            self.config["nominal_optimization"].get(
                "position_constraint_scale_km", 1000.0
            )
        )
        return np.asarray(result["position_error_km"], dtype=float) / scale

    def dv2_limit_constraint(self, x: np.ndarray) -> float:
        result = self.evaluate(x)
        maximum = self.config["nominal_optimization"].get("dv2_max_mps", None)
        if maximum is None:
            return 1.0
        return float(maximum) - float(result["dv2_mag_mps"])

    def dv1_norm_limit_constraint(self, x: np.ndarray) -> float:
        result = self.evaluate(x)
        maximum = self.config["nominal_optimization"].get("dv1_max_mps", None)
        if maximum is None:
            return 1.0
        return float(maximum) - float(result["dv1_mag_mps"])


def nominal_variable_bounds(config: dict[str, Any]) -> list[tuple[float, float]]:
    separation_cfg = config["separation"]
    optimization_cfg = config["nominal_optimization"]
    alpha_bounds = separation_cfg["azimuth_bounds_deg"]
    beta_bounds = separation_cfg["elevation_bounds_deg"]
    component_bound = float(optimization_cfg["dv1_component_bound_mps"])
    return [
        (np.deg2rad(float(alpha_bounds["min"])), np.deg2rad(float(alpha_bounds["max"]))),
        (np.deg2rad(float(beta_bounds["min"])), np.deg2rad(float(beta_bounds["max"]))),
        (-component_bound, component_bound),
        (-component_bound, component_bound),
        (-component_bound, component_bound),
    ]


def fixed_angle_pretarget(
    evaluator: NominalFiveVariableEvaluator,
    alpha_rad: float,
    beta_rad: float,
    dv1_initial_mps: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    optimization_cfg = config["nominal_optimization"]
    if not as_bool(optimization_cfg.get("pretarget_dv1", True)):
        return np.asarray(dv1_initial_mps, dtype=float)

    component_bound = float(optimization_cfg["dv1_component_bound_mps"])
    scale = float(optimization_cfg.get("position_constraint_scale_km", 1000.0))

    def residual(dv1_mps: np.ndarray) -> np.ndarray:
        x = np.hstack([alpha_rad, beta_rad, dv1_mps])
        return evaluator.evaluate(x)["position_error_km"] / scale

    result = least_squares(
        residual,
        np.asarray(dv1_initial_mps, dtype=float),
        bounds=(
            np.full(3, -component_bound, dtype=float),
            np.full(3, component_bound, dtype=float),
        ),
        max_nfev=int(optimization_cfg.get("pretarget_max_nfev", 100)),
        xtol=float(optimization_cfg.get("pretarget_xtol", 1.0e-9)),
        ftol=float(optimization_cfg.get("pretarget_ftol", 1.0e-9)),
        gtol=float(optimization_cfg.get("pretarget_gtol", 1.0e-9)),
        verbose=0,
    )
    return np.asarray(result.x, dtype=float)


def nominal_multistart_guesses(
    config: dict[str, Any],
    task_seed: int,
) -> list[np.ndarray]:
    optimization_cfg = config["nominal_optimization"]
    multistart_cfg = optimization_cfg.get("multistart", {})
    initial_dv_cfg = optimization_cfg.get("dv1_initial_guess_mps", {})
    default_dv = np.array(
        [
            float(initial_dv_cfg.get("x", 0.0)),
            float(initial_dv_cfg.get("y", 0.0)),
            float(initial_dv_cfg.get("z", 0.0)),
        ],
        dtype=float,
    )

    starts: list[np.ndarray] = []
    # An explicitly empty list now genuinely disables explicit orientation
    # starts. The six defaults are used only when the key is omitted entirely.
    if "explicit_angle_starts_deg" in multistart_cfg:
        explicit_starts = multistart_cfg["explicit_angle_starts_deg"] or []
    else:
        explicit_starts = [
            {"azimuth": 0.0, "elevation": 0.0},
            {"azimuth": 180.0, "elevation": 0.0},
            {"azimuth": 90.0, "elevation": 0.0},
            {"azimuth": -90.0, "elevation": 0.0},
            {"azimuth": 0.0, "elevation": 45.0},
            {"azimuth": 0.0, "elevation": -45.0},
        ]
    for item in explicit_starts:
        starts.append(
            np.hstack(
                [
                    np.deg2rad(float(item["azimuth"])),
                    np.deg2rad(float(item["elevation"])),
                    default_dv,
                ]
            )
        )

    n_random = int(multistart_cfg.get("n_random_starts", 12))
    if n_random > 0:
        rng = np.random.default_rng(int(task_seed))
        bounds = nominal_variable_bounds(config)
        for _ in range(n_random):
            alpha = rng.uniform(bounds[0][0], bounds[0][1])
            # Uniform sphere sampling within configured beta range is not exact
            # when bounds are restricted, but a uniform sine(elevation) draw is
            # considerably better conditioned than uniform elevation.
            sin_low = math.sin(bounds[1][0])
            sin_high = math.sin(bounds[1][1])
            beta = math.asin(rng.uniform(sin_low, sin_high))
            starts.append(np.hstack([alpha, beta, default_dv]))

    # Remove duplicate starts while preserving order.
    unique: list[np.ndarray] = []
    for start in starts:
        if not any(np.allclose(start, existing, atol=1.0e-12, rtol=0.0) for existing in unique):
            unique.append(start)
    return unique


def optimize_nominal_wait_case(
    task: pd.Series,
    phase: dict[str, Any],
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    config: dict[str, Any],
) -> dict[str, Any]:
    case_start = time.time()
    deadline = make_case_deadline(config, case_start)
    wait_days = float(task["wait_after_commissioning_days"])
    first_burn_jdtdb = (
        reference.start_jdtdb
        + float(phase["commissioning_duration_days"])
        + wait_days
    )
    acquisition_jdtdb = float(phase["acquisition_jdtdb"])
    if first_burn_jdtdb >= acquisition_jdtdb:
        raise TemporalInfeasibilityError(
            "First burn is not before acquisition for this wait candidate."
        )

    evaluator = NominalFiveVariableEvaluator(
        propagator=propagator,
        mothership_state_eme=reference.deployment_state_eme,
        deployment_jdtdb=reference.start_jdtdb,
        first_burn_jdtdb=first_burn_jdtdb,
        acquisition_jdtdb=acquisition_jdtdb,
        target_state_eme=phase["target_state_eme"],
        local_basis_secr=reference.deployment_local_basis_secr,
        fixed_separation_speed_mps=float(config["separation"]["fixed_nominal_speed_mps"]),
        config=config,
        deadline=deadline,
    )

    bounds = nominal_variable_bounds(config)
    optimization_cfg = config["nominal_optimization"]
    fixed_angles = bool(
        np.isclose(bounds[0][0], bounds[0][1], rtol=0.0, atol=1.0e-15)
        and np.isclose(bounds[1][0], bounds[1][1], rtol=0.0, atol=1.0e-15)
    )
    use_fixed_angle_fast_path = bool(
        fixed_angles and as_bool(optimization_cfg.get("fixed_angle_fast_path", True))
    )

    candidate_records: list[dict[str, Any]] = []
    if use_fixed_angle_fast_path:
        # With both angles fixed, the three dV1 components are the only unknowns
        # and the three terminal-position equations determine them. Running a
        # five-variable SLSQP after this solve only repeats expensive propagation.
        alpha_fixed = float(bounds[0][0])
        beta_fixed = float(bounds[1][0])
        initial_dv_cfg = optimization_cfg.get("dv1_initial_guess_mps", {})
        dv_initial = np.array(
            [
                float(initial_dv_cfg.get("x", 0.0)),
                float(initial_dv_cfg.get("y", 0.0)),
                float(initial_dv_cfg.get("z", 0.0)),
            ],
            dtype=float,
        )
        component_bound = float(optimization_cfg["dv1_component_bound_mps"])
        scale = float(optimization_cfg.get("position_constraint_scale_km", 1000.0))

        def residual(dv1_mps: np.ndarray) -> np.ndarray:
            check_deadline(deadline)
            x = np.hstack([alpha_fixed, beta_fixed, dv1_mps])
            return evaluator.evaluate(x)["position_error_km"] / scale

        result = least_squares(
            residual,
            dv_initial,
            bounds=(
                np.full(3, -component_bound, dtype=float),
                np.full(3, component_bound, dtype=float),
            ),
            max_nfev=int(optimization_cfg.get("pretarget_max_nfev", 100)),
            xtol=float(optimization_cfg.get("pretarget_xtol", 1.0e-9)),
            ftol=float(optimization_cfg.get("pretarget_ftol", 1.0e-9)),
            gtol=float(optimization_cfg.get("pretarget_gtol", 1.0e-9)),
            verbose=0,
        )
        evaluated = evaluator.evaluate(
            np.hstack([alpha_fixed, beta_fixed, np.asarray(result.x, dtype=float)])
        )
        position_error_norm = float(np.linalg.norm(evaluated["position_error_km"]))
        velocity_error_norm = float(np.linalg.norm(evaluated["velocity_error_mps"]))
        position_tolerance = float(
            optimization_cfg.get("position_success_tolerance_km", 1.0)
        )
        velocity_tolerance = float(
            optimization_cfg.get("velocity_success_tolerance_mps", 1.0e-3)
        )
        norm_limits_satisfied = True
        if optimization_cfg.get("dv1_max_mps", None) is not None:
            norm_limits_satisfied &= evaluated["dv1_mag_mps"] <= float(
                optimization_cfg["dv1_max_mps"]
            )
        if optimization_cfg.get("dv2_max_mps", None) is not None:
            norm_limits_satisfied &= evaluated["dv2_mag_mps"] <= float(
                optimization_cfg["dv2_max_mps"]
            )
        target_success = bool(
            position_error_norm <= position_tolerance
            and velocity_error_norm <= velocity_tolerance
            and norm_limits_satisfied
        )
        candidate_records.append(
            {
                "start_index": 0,
                "result": result,
                "evaluated": evaluated,
                "optimizer_success": bool(result.success),
                "target_success": target_success,
                "position_error_norm_km": position_error_norm,
                "velocity_error_norm_mps": velocity_error_norm,
            }
        )
    else:
        task_seed = int(optimization_cfg.get("multistart", {}).get("random_seed", 1))
        task_seed += int(task["phase_index"]) * 1_000_003 + int(task["wait_index"]) * 10_007
        if str(task["stage"]) == "refine":
            task_seed += 900_001
        starts = nominal_multistart_guesses(config, task_seed)
        if not starts:
            raise ValueError(
                "No nominal optimization starts were configured. Provide at least "
                "one explicit start or set n_random_starts > 0."
            )

        for start_index, start in enumerate(starts):
            check_deadline(deadline)
            alpha_start = float(start[0])
            beta_start = float(start[1])
            dv_start = fixed_angle_pretarget(
                evaluator,
                alpha_start,
                beta_start,
                np.asarray(start[2:5], dtype=float),
                config,
            )
            x0 = np.hstack([alpha_start, beta_start, dv_start])

            constraints: list[dict[str, Any]] = [
                {"type": "eq", "fun": evaluator.position_constraint}
            ]
            if optimization_cfg.get("dv2_max_mps", None) is not None:
                constraints.append({"type": "ineq", "fun": evaluator.dv2_limit_constraint})
            if optimization_cfg.get("dv1_max_mps", None) is not None:
                constraints.append({"type": "ineq", "fun": evaluator.dv1_norm_limit_constraint})

            result = minimize(
                evaluator.objective,
                x0,
                method=str(optimization_cfg.get("method", "SLSQP")),
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": int(optimization_cfg.get("max_iterations", 500)),
                    "ftol": float(optimization_cfg.get("ftol", 1.0e-10)),
                    "disp": False,
                },
            )
            evaluated = evaluator.evaluate(result.x)
            position_error_norm = float(np.linalg.norm(evaluated["position_error_km"]))
            velocity_error_norm = float(np.linalg.norm(evaluated["velocity_error_mps"]))
            position_tolerance = float(
                optimization_cfg.get("position_success_tolerance_km", 1.0)
            )
            velocity_tolerance = float(
                optimization_cfg.get("velocity_success_tolerance_mps", 1.0e-3)
            )
            target_success = bool(
                position_error_norm <= position_tolerance
                and velocity_error_norm <= velocity_tolerance
            )
            candidate_records.append(
                {
                    "start_index": start_index,
                    "result": result,
                    "evaluated": evaluated,
                    "optimizer_success": bool(result.success),
                    "target_success": target_success,
                    "position_error_norm_km": position_error_norm,
                    "velocity_error_norm_mps": velocity_error_norm,
                }
            )

    feasible = [record for record in candidate_records if record["target_success"]]
    if feasible:
        selected = min(feasible, key=lambda item: item["evaluated"]["dv_total_mps"])
    else:
        selected = min(
            candidate_records,
            key=lambda item: (
                item["position_error_norm_km"],
                item["evaluated"]["dv_total_mps"],
            ),
        )

    result = selected["result"]
    evaluated = selected["evaluated"]
    target_state = np.asarray(phase["target_state_eme"], dtype=float)
    final_state = evaluated["state_acquisition_plus"]
    record: dict[str, Any] = {
        "task_id": str(task["task_id"]),
        "stage": str(task["stage"]),
        "phase_index": int(task["phase_index"]),
        "separation_id": str(task["separation_id"]),
        "separation_phase_rad": float(phase["separation_phase_rad"]),
        "separation_phase_deg": float(phase["separation_phase_deg"]),
        "separation_phase_fraction": float(phase["separation_phase_fraction"]),
        "reference_period_days": float(phase["reference_period_days"]),
        "deployment_jdtdb": float(reference.start_jdtdb),
        "deployment_utc": jdtdb_to_utc(reference.start_jdtdb),
        "commissioning_duration_days": float(phase["commissioning_duration_days"]),
        "total_deployment_to_acquisition_days": float(
            phase["total_deployment_to_acquisition_days"]
        ),
        "wait_after_commissioning_days": wait_days,
        "wait_fraction_within_configured_sweep": float(
            task.get("wait_fraction_within_configured_sweep", np.nan)
        ),
        "first_burn_jdtdb": float(first_burn_jdtdb),
        "first_burn_utc": jdtdb_to_utc(first_burn_jdtdb),
        "post_first_burn_transfer_days": float(acquisition_jdtdb - first_burn_jdtdb),
        "acquisition_jdtdb": acquisition_jdtdb,
        "acquisition_utc": jdtdb_to_utc(acquisition_jdtdb),
        "fixed_separation_speed_mps": float(
            config["separation"]["fixed_nominal_speed_mps"]
        ),
        "optimized_separation_azimuth_rad": float(evaluated["alpha_rad"]),
        "optimized_separation_azimuth_deg": float(np.rad2deg(evaluated["alpha_rad"])),
        "optimized_separation_elevation_rad": float(evaluated["beta_rad"]),
        "optimized_separation_elevation_deg": float(np.rad2deg(evaluated["beta_rad"])),
        "separation_dv_x_geo_eme_mps": float(1000.0 * evaluated["separation_dv_eme_kms"][0]),
        "separation_dv_y_geo_eme_mps": float(1000.0 * evaluated["separation_dv_eme_kms"][1]),
        "separation_dv_z_geo_eme_mps": float(1000.0 * evaluated["separation_dv_eme_kms"][2]),
        "dv1_x_mps": float(evaluated["dv1_mps"][0]),
        "dv1_y_mps": float(evaluated["dv1_mps"][1]),
        "dv1_z_mps": float(evaluated["dv1_mps"][2]),
        "dv1_mag_mps": float(evaluated["dv1_mag_mps"]),
        "dv2_x_mps": float(evaluated["dv2_mps"][0]),
        "dv2_y_mps": float(evaluated["dv2_mps"][1]),
        "dv2_z_mps": float(evaluated["dv2_mps"][2]),
        "dv2_mag_mps": float(evaluated["dv2_mag_mps"]),
        "dv_total_mps": float(evaluated["dv_total_mps"]),
        "optimizer_success": bool(result.success),
        "target_success": bool(selected["target_success"]),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(getattr(result, "nit", -1)),
        "optimizer_function_evaluations": int(getattr(result, "nfev", -1)),
        "selected_multistart_index": int(selected["start_index"]),
        "n_multistarts_attempted": int(len(candidate_records)),
        "fixed_angle_fast_path_used": bool(use_fixed_angle_fast_path),
        "unique_trajectory_evaluations": int(evaluator.unique_trajectory_evaluations),
        "preburn_propagations": int(evaluator.preburn_propagations),
        "postburn_propagations": int(evaluator.postburn_propagations),
        "optimization_runtime_seconds": float(time.time() - case_start),
        "optimization_timed_out": False,
        "final_position_error_x_km": float(evaluated["position_error_km"][0]),
        "final_position_error_y_km": float(evaluated["position_error_km"][1]),
        "final_position_error_z_km": float(evaluated["position_error_km"][2]),
        "final_position_error_norm_km": float(selected["position_error_norm_km"]),
        "final_velocity_error_x_mps": float(evaluated["velocity_error_mps"][0]),
        "final_velocity_error_y_mps": float(evaluated["velocity_error_mps"][1]),
        "final_velocity_error_z_mps": float(evaluated["velocity_error_mps"][2]),
        "final_velocity_error_norm_mps": float(selected["velocity_error_norm_mps"]),
    }

    for prefix, state in [
        ("initial", evaluated["state0"]),
        ("target", target_state),
        ("final", final_state),
    ]:
        record.update(
            {
                f"{prefix}_x_geo_eme_km": float(state[0]),
                f"{prefix}_y_geo_eme_km": float(state[1]),
                f"{prefix}_z_geo_eme_km": float(state[2]),
                f"{prefix}_vx_geo_eme_kms": float(state[3]),
                f"{prefix}_vy_geo_eme_kms": float(state[4]),
                f"{prefix}_vz_geo_eme_kms": float(state[5]),
            }
        )

    for key, value in phase.items():
        if key.endswith("_error_km") or key.endswith("_error_mps"):
            record[key] = value

    return record


def failed_record(task: pd.Series, exception: Exception) -> dict[str, Any]:
    timed_out = isinstance(exception, OptimizationTimeoutError)
    return {
        "task_id": str(task.get("task_id", "")),
        "stage": str(task.get("stage", "")),
        "phase_index": int(task.get("phase_index", -1)),
        "separation_id": str(task.get("separation_id", "")),
        "separation_phase_rad": float(task.get("separation_phase_rad", np.nan)),
        "wait_after_commissioning_days": float(
            task.get("wait_after_commissioning_days", np.nan)
        ),
        "optimizer_success": False,
        "target_success": False,
        "optimization_timed_out": timed_out,
        "optimizer_status": -998 if timed_out else -999,
        "optimizer_message": str(exception),
        "dv_total_mps": np.nan,
        "final_position_error_norm_km": np.nan,
        "final_velocity_error_norm_mps": np.nan,
    }


# =============================================================================
# MPI task execution and rank outputs
# =============================================================================


def save_rank_records(
    output_dir: Path,
    mode: str,
    stage: str,
    rank: int,
    records: list[dict[str, Any]],
    enabled: bool,
) -> Optional[Path]:
    if not enabled:
        return None
    rank_dir = output_dir / "rank_outputs"
    rank_dir.mkdir(parents=True, exist_ok=True)
    final_path = rank_dir / f"{mode}_{stage}_rank_{rank:04d}.csv"
    temporary_path = final_path.with_suffix(".tmp.csv")
    pd.DataFrame(records).to_csv(temporary_path, index=False)
    temporary_path.replace(final_path)
    return final_path


def run_nominal_task_stage(
    tasks: pd.DataFrame,
    stage: str,
    phase_contexts: list[dict[str, Any]],
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    config: dict[str, Any],
    comm: MPI.Comm,
    run_start_time: float,
    output_dir: Path,
) -> tuple[Optional[pd.DataFrame], list[bool], list[int], list[int]]:
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_tasks = tasks.iloc[rank::size].copy()
    local_records: list[dict[str, Any]] = []
    stopped = False
    progress_every = int(config.get("mpi", {}).get("progress_every", 1))

    for local_index, (_, task) in enumerate(local_tasks.iterrows()):
        if should_save_and_stop(config, run_start_time):
            stopped = True
            break
        if progress_every > 0 and (
            local_index == 0 or (local_index + 1) % progress_every == 0
        ):
            print(
                f"[rank {rank:04d}/{size:04d}] {stage} task "
                f"{local_index + 1}/{len(local_tasks)}: {task['task_id']}",
                flush=True,
            )
        phase = phase_contexts[int(task["phase_index"])]
        try:
            record = optimize_nominal_wait_case(
                task, phase, reference, propagator, config
            )
        except Exception as exception:  # continue independent MPI cases
            record = failed_record(task, exception)
        local_records.append(record)

    save_rank_records(
        output_dir,
        mode="nominal",
        stage=stage,
        rank=rank,
        records=local_records,
        enabled=as_bool(config.get("mpi", {}).get("save_rank_outputs", True)),
    )

    stopped_flags = comm.gather(stopped, root=0)
    assigned_counts = comm.gather(len(local_tasks), root=0)
    completed_counts = comm.gather(len(local_records), root=0)
    gathered = comm.gather(local_records, root=0)

    if rank != 0:
        return None, [], [], []
    records = [record for block in gathered for record in block]
    return pd.DataFrame(records), stopped_flags, assigned_counts, completed_counts


# =============================================================================
# Nominal result selection, trajectories, and plots
# =============================================================================


def select_nominal_phase_results(wait_results: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    for separation_id, group in wait_results.groupby("separation_id", sort=False):
        successful = successful_wait_subset(group)
        if not successful.empty:
            selected = successful.loc[successful["dv_total_mps"].idxmin()].copy()
        else:
            finite = group.dropna(subset=["final_position_error_norm_km"])
            if finite.empty:
                selected = group.iloc[0].copy()
            else:
                selected = finite.loc[finite["final_position_error_norm_km"].idxmin()].copy()
        selected["selected_as_nominal"] = True
        selected_rows.append(selected)
    if not selected_rows:
        return pd.DataFrame()
    return pd.DataFrame(selected_rows).sort_values("separation_phase_rad").reset_index(drop=True)


def propagate_impulsive_trajectory(
    propagator: NBodyPropagator,
    state0: np.ndarray,
    t0: float,
    t_burn: float,
    t_acquisition: float,
    dv1_mps: np.ndarray,
    dv2_mps: np.ndarray,
    step_days: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    times_pre = build_time_grid(t0, t_burn, step_days)
    states_pre = propagate_grid(propagator, state0, t0, times_pre)
    state_burn_plus = states_pre[-1].copy()
    state_burn_plus[3:] += np.asarray(dv1_mps, dtype=float) / 1000.0
    states_pre[-1] = state_burn_plus

    times_post = build_time_grid(t_burn, t_acquisition, step_days)
    states_post = propagate_grid(propagator, state_burn_plus, t_burn, times_post)
    states_post[-1, 3:] += np.asarray(dv2_mps, dtype=float) / 1000.0

    times = np.concatenate([times_pre, times_post[1:]])
    states = np.vstack([states_pre, states_post[1:]])
    events = [""] * len(times)
    burn_index = int(np.argmin(np.abs(times - t_burn)))
    acquisition_index = int(np.argmin(np.abs(times - t_acquisition)))
    events[0] = "SEPARATION"
    events[burn_index] = "CORRECTION_BURN_POST"
    events[acquisition_index] = "ACQUISITION_BURN_POST"
    # The sampled state at each maneuver epoch is the post-burn state. The
    # maneuver file separately preserves exact minus and plus states.
    return times, states, events


def mothership_reference_states(
    propagator: NBodyPropagator,
    reference: ReferenceOrbit,
    times: np.ndarray,
) -> np.ndarray:
    return propagate_grid(
        propagator,
        reference.deployment_state_eme,
        reference.start_jdtdb,
        np.asarray(times, dtype=float),
    )


def phase_zero_reference_states(reference: ReferenceOrbit, times: np.ndarray) -> np.ndarray:
    states = [state_secr_to_eme(reference.phase_zero_state_secr, float(t)) for t in times]
    return np.vstack(states)


def trajectory_bundle_dataframe(
    times: np.ndarray,
    microsat_states: np.ndarray,
    mothership_states: np.ndarray,
    phase_reference_states: np.ndarray,
    events: list[str],
    nominal_deployment_jdtdb: float,
) -> pd.DataFrame:
    relative_r = microsat_states[:, :3] - mothership_states[:, :3]
    relative_v_mps = 1000.0 * (microsat_states[:, 3:] - mothership_states[:, 3:])
    phase_error_r = microsat_states[:, :3] - phase_reference_states[:, :3]
    phase_error_v_mps = 1000.0 * (
        microsat_states[:, 3:] - phase_reference_states[:, 3:]
    )

    data: dict[str, Any] = {
        "jdtdb": times,
        "utc": [jdtdb_to_utc(float(t)) for t in times],
        "t_days_since_nominal_deployment": times - float(nominal_deployment_jdtdb),
        "event": events,
    }

    names = ["x", "y", "z", "vx", "vy", "vz"]
    units = ["km", "km", "km", "kms", "kms", "kms"]
    for prefix, states in [
        ("mothership_ref", mothership_states),
        ("microsat_phase_ref", phase_reference_states),
        ("microsat_actual", microsat_states),
    ]:
        for index, (name, unit_name) in enumerate(zip(names, units)):
            data[f"{prefix}_{name}_geo_eme_{unit_name}"] = states[:, index]

    for index, axis in enumerate(["x", "y", "z"]):
        data[f"relative_{axis}_km"] = relative_r[:, index]
        data[f"relative_v{axis}_mps"] = relative_v_mps[:, index]
        data[f"phase_reference_error_{axis}_km"] = phase_error_r[:, index]
        data[f"phase_reference_error_v{axis}_mps"] = phase_error_v_mps[:, index]
    data["relative_distance_km"] = np.linalg.norm(relative_r, axis=1)
    data["relative_speed_mps"] = np.linalg.norm(relative_v_mps, axis=1)
    data["phase_reference_position_error_km"] = np.linalg.norm(phase_error_r, axis=1)
    data["phase_reference_velocity_error_mps"] = np.linalg.norm(
        phase_error_v_mps, axis=1
    )
    return pd.DataFrame(data)


def save_maneuver_file(
    path: Path,
    row: pd.Series | dict[str, Any],
    propagator: NBodyPropagator,
) -> None:
    row = pd.Series(row)
    state0 = np.array(
        [
            row["initial_x_geo_eme_km"],
            row["initial_y_geo_eme_km"],
            row["initial_z_geo_eme_km"],
            row["initial_vx_geo_eme_kms"],
            row["initial_vy_geo_eme_kms"],
            row["initial_vz_geo_eme_kms"],
        ],
        dtype=float,
    )
    t0 = float(row.get("actual_deployment_jdtdb", row["deployment_jdtdb"]))
    t1 = float(row.get("actual_first_burn_jdtdb", row["first_burn_jdtdb"]))
    ta = float(row["acquisition_jdtdb"])
    dv1 = np.array([row["dv1_x_mps"], row["dv1_y_mps"], row["dv1_z_mps"]])
    dv2 = np.array([row["dv2_x_mps"], row["dv2_y_mps"], row["dv2_z_mps"]])
    state1_minus = propagate_state(propagator, state0, t0, t1)
    state1_plus = state1_minus.copy()
    state1_plus[3:] += dv1 / 1000.0
    state2_minus = propagate_state(propagator, state1_plus, t1, ta)
    state2_plus = state2_minus.copy()
    state2_plus[3:] += dv2 / 1000.0

    records = []
    for name, epoch, delta_v, state_minus, state_plus in [
        ("CORRECTION", t1, dv1, state1_minus, state1_plus),
        ("ACQUISITION", ta, dv2, state2_minus, state2_plus),
    ]:
        record: dict[str, Any] = {
            "maneuver": name,
            "jdtdb": epoch,
            "utc": jdtdb_to_utc(epoch),
            "dv_x_mps": delta_v[0],
            "dv_y_mps": delta_v[1],
            "dv_z_mps": delta_v[2],
            "dv_mag_mps": float(np.linalg.norm(delta_v)),
        }
        for side, state in [("minus", state_minus), ("plus", state_plus)]:
            record.update(
                {
                    f"state_{side}_x_km": state[0],
                    f"state_{side}_y_km": state[1],
                    f"state_{side}_z_km": state[2],
                    f"state_{side}_vx_kms": state[3],
                    f"state_{side}_vy_kms": state[4],
                    f"state_{side}_vz_kms": state[5],
                }
            )
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def plot_configuration(config: dict[str, Any]) -> tuple[tuple[float, float], float, str]:
    plot_cfg = config.get("plots", {})
    dimensions = plot_cfg.get("figure_size_inches", {})
    size = (
        float(dimensions.get("width", 6.0)),
        float(dimensions.get("height", 6.0)),
    )
    font_size = float(plot_cfg.get("font_size", 12.0))
    extension = str(plot_cfg.get("format", "svg")).lstrip(".")
    return size, font_size, extension


def apply_plot_font(config: dict[str, Any]) -> None:
    _, font_size, _ = plot_configuration(config)
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )


def save_wait_sweep_plot(
    group: pd.DataFrame,
    selected_wait: float,
    path: Path,
    config: dict[str, Any],
) -> None:
    size, _, _ = plot_configuration(config)
    fig, ax = plt.subplots(figsize=size)
    successful = successful_wait_subset(group).sort_values("wait_after_commissioning_days")
    if not successful.empty:
        ax.plot(
            successful["wait_after_commissioning_days"],
            successful["dv_total_mps"],
            marker="o",
        )
        selected_row = successful.iloc[
            int(
                np.argmin(
                    np.abs(
                        successful["wait_after_commissioning_days"].to_numpy(dtype=float)
                        - selected_wait
                    )
                )
            )
        ]
        ax.scatter(
            [selected_row["wait_after_commissioning_days"]],
            [selected_row["dv_total_mps"]],
            marker="*",
            s=120,
            label="Selected nominal",
        )
        ax.legend()
    ax.set_xlabel("Wait after commissioning [days]")
    ax.set_ylabel("Corrective $\\Delta V$ [m/s]")
    ax.set_title("Wait-time trade")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_nominal_summary_plots(
    nominal_df: pd.DataFrame,
    plot_dir: Path,
    config: dict[str, Any],
) -> None:
    if nominal_df.empty or not as_bool(config.get("plots", {}).get("enabled", True)):
        return
    size, _, extension = plot_configuration(config)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        (
            "dv_total_mps",
            "Corrective $\\Delta V$ [m/s]",
            "nominal_dv_vs_phase",
        ),
        (
            "wait_after_commissioning_days",
            "Optimal wait after commissioning [days]",
            "nominal_wait_vs_phase",
        ),
        (
            "optimized_separation_azimuth_deg",
            "Optimized separation azimuth [deg]",
            "nominal_azimuth_vs_phase",
        ),
        (
            "optimized_separation_elevation_deg",
            "Optimized separation elevation [deg]",
            "nominal_elevation_vs_phase",
        ),
    ]
    for column, ylabel, filename in plot_specs:
        if column not in nominal_df.columns:
            continue
        fig, ax = plt.subplots(figsize=size)
        ax.plot(nominal_df["separation_phase_deg"], nominal_df[column], marker="o")
        ax.set_xlabel("Desired mothership phase separation [deg]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{filename}.{extension}", bbox_inches="tight")
        plt.close(fig)


def save_nominal_trajectory_and_files(
    row: pd.Series,
    all_wait_results: pd.DataFrame,
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[str, str, str]:
    phase_dir = output_dir / "nominal" / safe_filename(str(row["separation_id"]))
    phase_dir.mkdir(parents=True, exist_ok=True)
    wait_path = phase_dir / "wait_time_sweep.csv"

    # Always preserve the wait-sweep table, including when every optimization
    # for this separation failed or timed out.
    group = all_wait_results[
        all_wait_results["separation_id"] == row["separation_id"]
    ].copy()
    group = group.sort_values(["wait_after_commissioning_days", "stage"])
    group["selected_as_nominal"] = False
    if (
        "wait_after_commissioning_days" in row.index
        and pd.notna(row.get("wait_after_commissioning_days", np.nan))
        and "target_success" in group.columns
    ):
        group["selected_as_nominal"] = (
            np.isclose(
                group["wait_after_commissioning_days"].to_numpy(dtype=float),
                float(row["wait_after_commissioning_days"]),
                atol=1.0e-12,
                rtol=0.0,
            )
            & group["target_success"].fillna(False).astype(bool).to_numpy()
        )
    group.to_csv(wait_path, index=False)

    if (
        as_bool(config.get("plots", {}).get("enabled", True))
        and as_bool(config.get("plots", {}).get("save_wait_time_trade_curves", True))
        and pd.notna(row.get("wait_after_commissioning_days", np.nan))
    ):
        _, _, extension = plot_configuration(config)
        save_wait_sweep_plot(
            group,
            float(row["wait_after_commissioning_days"]),
            phase_dir / f"wait_time_trade.{extension}",
            config,
        )

    required_columns = [
        "initial_x_geo_eme_km",
        "initial_y_geo_eme_km",
        "initial_z_geo_eme_km",
        "initial_vx_geo_eme_kms",
        "initial_vy_geo_eme_kms",
        "initial_vz_geo_eme_kms",
        "dv1_x_mps",
        "dv1_y_mps",
        "dv1_z_mps",
        "dv2_x_mps",
        "dv2_y_mps",
        "dv2_z_mps",
        "deployment_jdtdb",
        "first_burn_jdtdb",
        "acquisition_jdtdb",
    ]
    has_complete_solution = bool(
        row.get("target_success", False)
        and all(column in row.index for column in required_columns)
        and all(pd.notna(row[column]) for column in required_columns)
    )
    if not has_complete_solution:
        # A failed fallback row does not contain propagated state fields. Return
        # blank trajectory paths instead of raising KeyError.
        return "", "", str(wait_path)

    step_days = float(
        config.get("output", {}).get("trajectories", {}).get("step_days", 0.05)
    )
    state0 = np.array(
        [
            row["initial_x_geo_eme_km"],
            row["initial_y_geo_eme_km"],
            row["initial_z_geo_eme_km"],
            row["initial_vx_geo_eme_kms"],
            row["initial_vy_geo_eme_kms"],
            row["initial_vz_geo_eme_kms"],
        ],
        dtype=float,
    )
    dv1 = np.array([row["dv1_x_mps"], row["dv1_y_mps"], row["dv1_z_mps"]])
    dv2 = np.array([row["dv2_x_mps"], row["dv2_y_mps"], row["dv2_z_mps"]])
    times, microsat_states, events = propagate_impulsive_trajectory(
        propagator,
        state0,
        float(row["deployment_jdtdb"]),
        float(row["first_burn_jdtdb"]),
        float(row["acquisition_jdtdb"]),
        dv1,
        dv2,
        step_days,
    )
    mothership_states = mothership_reference_states(propagator, reference, times)
    phase_reference_states = phase_zero_reference_states(reference, times)
    bundle = trajectory_bundle_dataframe(
        times,
        microsat_states,
        mothership_states,
        phase_reference_states,
        events,
        reference.start_jdtdb,
    )
    trajectory_path = phase_dir / "trajectory_bundle.csv"
    maneuver_path = phase_dir / "maneuvers.csv"
    bundle.to_csv(trajectory_path, index=False)
    save_maneuver_file(maneuver_path, row, propagator)
    return str(trajectory_path), str(maneuver_path), str(wait_path)


# =============================================================================
# Dispersion generation
# =============================================================================


def draw_normal(
    rng: np.random.Generator,
    sigma: np.ndarray | float,
    clip_sigma: Optional[float],
) -> np.ndarray:
    sigma_array = np.asarray(sigma, dtype=float)
    values = rng.normal(0.0, sigma_array)
    if clip_sigma is not None:
        limits = float(clip_sigma) * sigma_array
        values = np.clip(values, -limits, limits)
    return np.asarray(values, dtype=float)


def load_covariance_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    else:
        matrix = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError(f"Covariance matrix must be 6x6; got {matrix.shape}.")
    return matrix


def covariance_or_diagonal(
    block: dict[str, Any],
    position_unit_scale: float = 1.0,
    velocity_unit_scale: float = 1.0,
) -> np.ndarray:
    covariance_file = block.get("covariance_file", None)
    if covariance_file:
        return load_covariance_matrix(covariance_file)
    position = block.get("position_sigma_km", {})
    velocity = block.get("velocity_sigma_mps", {})
    sigmas = np.array(
        [
            float(position.get("x", 0.0)) * position_unit_scale,
            float(position.get("y", 0.0)) * position_unit_scale,
            float(position.get("z", 0.0)) * position_unit_scale,
            float(velocity.get("x", 0.0)) * velocity_unit_scale / 1000.0,
            float(velocity.get("y", 0.0)) * velocity_unit_scale / 1000.0,
            float(velocity.get("z", 0.0)) * velocity_unit_scale / 1000.0,
        ]
    )
    return np.diag(sigmas**2)


def draw_covariance(
    rng: np.random.Generator,
    covariance: np.ndarray,
    clip_sigma: Optional[float],
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    # Eigenvalue clipping makes a numerically semidefinite covariance usable.
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    z = rng.normal(size=6)
    if clip_sigma is not None:
        z = np.clip(z, -float(clip_sigma), float(clip_sigma))
    return factor @ z


def generate_dispersion_cases(
    nominal_row: pd.Series,
    reference: ReferenceOrbit,
    config: dict[str, Any],
) -> pd.DataFrame:
    dispersion_cfg = config["dispersion"]
    n_samples = int(dispersion_cfg["n_samples"])
    seed = int(dispersion_cfg.get("random_seed", 1))
    rng = np.random.default_rng(seed)
    clip_sigma_raw = dispersion_cfg.get("clip_sigma", None)
    clip_sigma = None if clip_sigma_raw is None else float(clip_sigma_raw)

    mothership_cfg = dispersion_cfg.get("mothership_state", {})
    mothership_covariance = covariance_or_diagonal(mothership_cfg)
    navigation_cfg = dispersion_cfg.get("navigation_at_first_burn", {})
    navigation_covariance = covariance_or_diagonal(navigation_cfg)
    acquisition_navigation_cfg = dispersion_cfg.get("navigation_at_acquisition", {})
    acquisition_navigation_covariance = covariance_or_diagonal(acquisition_navigation_cfg)

    rows: list[dict[str, Any]] = []
    for sample_index in range(n_samples):
        row: dict[str, Any] = {
            "dispersion_case_id": f"disp_{sample_index:06d}",
            "sample_id": sample_index,
            "random_seed": seed,
            "separation_id": str(nominal_row["separation_id"]),
            "separation_phase_rad": float(nominal_row["separation_phase_rad"]),
            "separation_phase_deg": float(nominal_row["separation_phase_deg"]),
        }

        deployment_timing_cfg = dispersion_cfg.get("deployment_timing", {})
        row["deployment_timing_error_s"] = (
            float(draw_normal(rng, float(deployment_timing_cfg.get("sigma_s", 0.0)), clip_sigma))
            if as_bool(deployment_timing_cfg.get("enabled", False))
            else 0.0
        )

        commissioning_cfg = config["commissioning"].get("duration_dispersion", {})
        commissioning_error_s = 0.0
        if as_bool(commissioning_cfg.get("enabled", False)):
            distribution = str(commissioning_cfg.get("distribution", "normal")).lower()
            sigma_s = 3600.0 * float(commissioning_cfg.get("sigma_hours", 0.0))
            if distribution in {"normal", "truncated_normal"}:
                commissioning_error_s = float(draw_normal(rng, sigma_s, clip_sigma))
            elif distribution == "uniform":
                half_width_s = 3600.0 * float(
                    commissioning_cfg.get("half_width_hours", 0.0)
                )
                commissioning_error_s = float(rng.uniform(-half_width_s, half_width_s))
            else:
                raise ValueError(
                    f"Unknown commissioning duration distribution: {distribution}"
                )
        row["commissioning_duration_error_s"] = commissioning_error_s

        if as_bool(mothership_cfg.get("enabled", False)):
            mothership_error = draw_covariance(rng, mothership_covariance, clip_sigma)
        else:
            mothership_error = np.zeros(6)
        for index, name in enumerate(["x", "y", "z"]):
            row[f"mothership_position_error_{name}_km"] = float(mothership_error[index])
            row[f"mothership_velocity_error_{name}_mps"] = float(
                1000.0 * mothership_error[index + 3]
            )

        separation_position_cfg = dispersion_cfg.get("separation_position", {})
        if as_bool(separation_position_cfg.get("enabled", False)):
            sigma_cfg = separation_position_cfg.get("sigma_km", {})
            separation_position = draw_normal(
                rng,
                np.array(
                    [
                        float(sigma_cfg.get("tangent", 0.0)),
                        float(sigma_cfg.get("cross", 0.0)),
                        float(sigma_cfg.get("normal", 0.0)),
                    ]
                ),
                clip_sigma,
            )
        else:
            separation_position = np.zeros(3)
        for index, name in enumerate(["tangent", "cross", "normal"]):
            row[f"separation_position_error_{name}_km"] = float(
                separation_position[index]
            )

        speed_cfg = dispersion_cfg.get("separation_speed", {})
        row["separation_speed_error_mps"] = (
            float(draw_normal(rng, float(speed_cfg.get("sigma_mps", 0.0)), clip_sigma))
            if as_bool(speed_cfg.get("enabled", False))
            else 0.0
        )

        direction_cfg = dispersion_cfg.get("separation_direction", {})
        if as_bool(direction_cfg.get("enabled", False)):
            row["separation_azimuth_error_deg"] = float(
                draw_normal(
                    rng,
                    float(direction_cfg.get("azimuth_sigma_deg", 0.0)),
                    clip_sigma,
                )
            )
            row["separation_elevation_error_deg"] = float(
                draw_normal(
                    rng,
                    float(direction_cfg.get("elevation_sigma_deg", 0.0)),
                    clip_sigma,
                )
            )
        else:
            row["separation_azimuth_error_deg"] = 0.0
            row["separation_elevation_error_deg"] = 0.0

        lever_cfg = dispersion_cfg.get("lever_arm", {})
        tipoff_sigma = lever_cfg.get("tipoff_rate_sigma_deg_s", {})
        if as_bool(lever_cfg.get("enabled", False)):
            tipoff = draw_normal(
                rng,
                np.deg2rad(
                    [
                        float(tipoff_sigma.get("x", 0.0)),
                        float(tipoff_sigma.get("y", 0.0)),
                        float(tipoff_sigma.get("z", 0.0)),
                    ]
                ),
                clip_sigma,
            )
        else:
            tipoff = np.zeros(3)
        for index, name in enumerate(["x", "y", "z"]):
            row[f"tipoff_rate_error_{name}_rad_s"] = float(tipoff[index])

        if as_bool(navigation_cfg.get("enabled", False)):
            navigation_error = draw_covariance(rng, navigation_covariance, clip_sigma)
        else:
            navigation_error = np.zeros(6)
        for index, name in enumerate(["x", "y", "z"]):
            row[f"first_burn_navigation_position_error_{name}_km"] = float(
                navigation_error[index]
            )
            row[f"first_burn_navigation_velocity_error_{name}_mps"] = float(
                1000.0 * navigation_error[index + 3]
            )

        if as_bool(acquisition_navigation_cfg.get("enabled", False)):
            acquisition_navigation_error = draw_covariance(
                rng, acquisition_navigation_covariance, clip_sigma
            )
        else:
            acquisition_navigation_error = np.zeros(6)
        for index, name in enumerate(["x", "y", "z"]):
            row[f"acquisition_navigation_position_error_{name}_km"] = float(
                acquisition_navigation_error[index]
            )
            row[f"acquisition_navigation_velocity_error_{name}_mps"] = float(
                1000.0 * acquisition_navigation_error[index + 3]
            )

        for burn_name in ["first_burn_execution", "acquisition_burn_execution"]:
            burn_cfg = dispersion_cfg.get(burn_name, {})
            prefix = "first_burn" if burn_name.startswith("first") else "acquisition_burn"
            if as_bool(burn_cfg.get("enabled", False)):
                row[f"{prefix}_magnitude_fraction_error"] = float(
                    draw_normal(
                        rng,
                        float(burn_cfg.get("magnitude_sigma_fraction", 0.0)),
                        clip_sigma,
                    )
                )
                pointing_sigma = float(burn_cfg.get("pointing_sigma_deg", 0.0))
                row[f"{prefix}_pointing_error_1_deg"] = float(
                    draw_normal(rng, pointing_sigma, clip_sigma)
                )
                row[f"{prefix}_pointing_error_2_deg"] = float(
                    draw_normal(rng, pointing_sigma, clip_sigma)
                )
                row[f"{prefix}_timing_error_s"] = float(
                    draw_normal(
                        rng,
                        float(burn_cfg.get("timing_sigma_s", 0.0)),
                        clip_sigma,
                    )
                )
            else:
                row[f"{prefix}_magnitude_fraction_error"] = 0.0
                row[f"{prefix}_pointing_error_1_deg"] = 0.0
                row[f"{prefix}_pointing_error_2_deg"] = 0.0
                row[f"{prefix}_timing_error_s"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Dispersion case construction and optimization
# =============================================================================


def row_vector(row: pd.Series, names: list[str], scale: float = 1.0) -> np.ndarray:
    return scale * np.array([float(row[name]) for name in names], dtype=float)


def perturb_burn_vector(
    command_mps: np.ndarray,
    magnitude_fraction_error: float,
    pointing_error_1_deg: float,
    pointing_error_2_deg: float,
) -> np.ndarray:
    command = np.asarray(command_mps, dtype=float)
    magnitude = float(np.linalg.norm(command))
    if magnitude <= 0.0:
        return command.copy()
    direction = unit(command)
    # Construct a stable orthogonal basis around the command direction.
    reference_axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(direction, reference_axis))) > 0.9:
        reference_axis = np.array([0.0, 1.0, 0.0])
    e1 = unit(np.cross(direction, reference_axis))
    e2 = unit(np.cross(direction, e1))
    angle1 = np.deg2rad(float(pointing_error_1_deg))
    angle2 = np.deg2rad(float(pointing_error_2_deg))
    perturbed_direction = unit(direction + angle1 * e1 + angle2 * e2)
    perturbed_magnitude = max(0.0, magnitude * (1.0 + float(magnitude_fraction_error)))
    return perturbed_magnitude * perturbed_direction


def actual_local_basis(
    nominal_mothership_state_eme: np.ndarray,
    jdtdb: float,
    reference: ReferenceOrbit,
    config: dict[str, Any],
) -> np.ndarray:
    epoch_mode = str(
        config.get("separation", {}).get(
            "direction_frame_epoch", "actual_deployment"
        )
    ).lower()
    if epoch_mode == "nominal_deployment":
        return reference.deployment_local_basis_secr
    if epoch_mode != "actual_deployment":
        raise ValueError(f"Unknown separation.direction_frame_epoch: {epoch_mode}")
    state_secr = state_eme_to_secr(nominal_mothership_state_eme, jdtdb)
    return build_local_quasi_halo_basis(state_secr)


def build_dispersed_initial_state(
    case: pd.Series,
    nominal_row: pd.Series,
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    config: dict[str, Any],
) -> dict[str, Any]:
    actual_deployment_jdtdb = reference.start_jdtdb + float(
        case["deployment_timing_error_s"]
    ) / 86400.0
    nominal_mothership_at_actual_epoch = propagate_state(
        propagator,
        reference.deployment_state_eme,
        reference.start_jdtdb,
        actual_deployment_jdtdb,
    )

    mothership_error = np.hstack(
        [
            row_vector(
                case,
                [
                    "mothership_position_error_x_km",
                    "mothership_position_error_y_km",
                    "mothership_position_error_z_km",
                ],
            ),
            row_vector(
                case,
                [
                    "mothership_velocity_error_x_mps",
                    "mothership_velocity_error_y_mps",
                    "mothership_velocity_error_z_mps",
                ],
                scale=1.0 / 1000.0,
            ),
        ]
    )
    actual_mothership_state = nominal_mothership_at_actual_epoch + mothership_error
    basis = actual_local_basis(
        nominal_mothership_at_actual_epoch,
        actual_deployment_jdtdb,
        reference,
        config,
    )

    position_local = row_vector(
        case,
        [
            "separation_position_error_tangent_km",
            "separation_position_error_cross_km",
            "separation_position_error_normal_km",
        ],
    )
    position_secr = basis @ position_local
    # Position conversion can be performed as a 3D position directly.
    earth_state = get_earth_heliocentric_eclip_state(actual_deployment_jdtdb)
    position_eclip = util.geo_secr_to_geo_eclip_generic(
        position_secr,
        earth_state,
        obj_hint=("position",),
        earth_hint=("state",),
    )
    position_eme = util.geo_eclip_to_geo_eme_generic(
        position_eclip, hint=("position",)
    )

    speed_mps = float(nominal_row["fixed_separation_speed_mps"]) + float(
        case["separation_speed_error_mps"]
    )
    speed_cfg = config["dispersion"].get("separation_speed", {})
    if as_bool(speed_cfg.get("prevent_negative_speed", True)):
        speed_mps = max(speed_mps, 0.0)
    alpha_rad = float(nominal_row["optimized_separation_azimuth_rad"]) + np.deg2rad(
        float(case["separation_azimuth_error_deg"])
    )
    beta_rad = float(nominal_row["optimized_separation_elevation_rad"]) + np.deg2rad(
        float(case["separation_elevation_error_deg"])
    )
    separation_dv_secr = local_spherical_vector(speed_mps, alpha_rad, beta_rad, basis)

    lever_cfg = config["dispersion"].get("lever_arm", {})
    lever_velocity_secr = np.zeros(3)
    if as_bool(lever_cfg.get("enabled", False)):
        position_body_m = lever_cfg.get("position_body_m", {})
        lever_position_km = np.array(
            [
                float(position_body_m.get("x", 0.0)),
                float(position_body_m.get("y", 0.0)),
                float(position_body_m.get("z", 0.0)),
            ],
            dtype=float,
        ) / 1000.0
        tipoff_rad_s = row_vector(
            case,
            [
                "tipoff_rate_error_x_rad_s",
                "tipoff_rate_error_y_rad_s",
                "tipoff_rate_error_z_rad_s",
            ],
        )
        # The configured body axes are treated as the optimized local deployment
        # frame for this preliminary translational coupling model.
        lever_velocity_local_kms = np.cross(tipoff_rad_s, lever_position_km)
        lever_velocity_secr = basis @ lever_velocity_local_kms

    separation_dv_eme = vector_secr_to_eme(
        separation_dv_secr + lever_velocity_secr,
        actual_deployment_jdtdb,
    )

    initial_state = actual_mothership_state.copy()
    initial_state[:3] += np.asarray(position_eme, dtype=float)
    initial_state[3:] += separation_dv_eme

    nominal_commissioning_days = float(config["commissioning"]["nominal_duration_days"])
    actual_commissioning_days = nominal_commissioning_days + float(
        case["commissioning_duration_error_s"]
    ) / 86400.0
    commissioning_dispersion_cfg = config["commissioning"].get(
        "duration_dispersion", {}
    )
    if as_bool(commissioning_dispersion_cfg.get("enabled", False)):
        minimum = commissioning_dispersion_cfg.get("minimum_duration_days", None)
        maximum = commissioning_dispersion_cfg.get("maximum_duration_days", None)
        if minimum is not None:
            actual_commissioning_days = max(actual_commissioning_days, float(minimum))
        if maximum is not None:
            actual_commissioning_days = min(actual_commissioning_days, float(maximum))
    if actual_commissioning_days < 0.0:
        raise TemporalInfeasibilityError("Actual commissioning duration is negative.")

    timing_mode = str(
        config.get("dispersion_optimization", {}).get(
            "correction_timing_mode", "fixed_wait_after_commissioning"
        )
    ).lower()
    nominal_wait_days = float(nominal_row["wait_after_commissioning_days"])
    if timing_mode == "fixed_wait_after_commissioning":
        scheduled_first_burn = (
            actual_deployment_jdtdb + actual_commissioning_days + nominal_wait_days
        )
    elif timing_mode == "fixed_absolute_epoch":
        scheduled_first_burn = float(nominal_row["first_burn_jdtdb"])
        commissioning_end = actual_deployment_jdtdb + actual_commissioning_days
        if commissioning_end > scheduled_first_burn:
            raise TemporalInfeasibilityError(
                "Commissioning completes after the fixed absolute correction epoch."
            )
    else:
        raise ValueError(
            "This implementation supports dispersion correction_timing_mode "
            "'fixed_wait_after_commissioning' or 'fixed_absolute_epoch'. "
            "Use a separate timing sensitivity study for per-case adaptive timing."
        )

    acquisition_jdtdb = float(nominal_row["acquisition_jdtdb"])
    if scheduled_first_burn >= acquisition_jdtdb:
        raise TemporalInfeasibilityError(
            "Scheduled first burn is not before the fixed acquisition epoch."
        )

    return {
        "actual_deployment_jdtdb": float(actual_deployment_jdtdb),
        "actual_deployment_utc": jdtdb_to_utc(actual_deployment_jdtdb),
        "nominal_mothership_state_at_actual_epoch": nominal_mothership_at_actual_epoch,
        "actual_mothership_state": actual_mothership_state,
        "local_basis_secr": basis,
        "actual_separation_speed_mps": speed_mps,
        "actual_separation_azimuth_rad": alpha_rad,
        "actual_separation_elevation_rad": beta_rad,
        "separation_dv_eme_kms": separation_dv_eme,
        "initial_state": initial_state,
        "actual_commissioning_duration_days": actual_commissioning_days,
        "scheduled_first_burn_jdtdb": float(scheduled_first_burn),
        "scheduled_first_burn_utc": jdtdb_to_utc(scheduled_first_burn),
        "remaining_transfer_duration_days": float(
            acquisition_jdtdb - scheduled_first_burn
        ),
    }


def optimize_dv1_for_position(
    propagator: NBodyPropagator,
    estimated_state_at_burn: np.ndarray,
    burn_jdtdb: float,
    acquisition_jdtdb: float,
    target_position_km: np.ndarray,
    initial_guess_mps: np.ndarray,
    config: dict[str, Any],
    deadline: Optional[float],
) -> tuple[np.ndarray, Any, np.ndarray]:
    optimization_cfg = config["dispersion_optimization"]
    scale = float(optimization_cfg.get("position_constraint_scale_km", 1000.0))
    component_bound = float(optimization_cfg["dv1_component_bound_mps"])
    cache_x: Optional[np.ndarray] = None
    cache_state: Optional[np.ndarray] = None

    def residual(dv1_mps: np.ndarray) -> np.ndarray:
        nonlocal cache_x, cache_state
        check_deadline(deadline)
        dv1_mps = np.asarray(dv1_mps, dtype=float)
        if cache_x is not None and np.array_equal(cache_x, dv1_mps):
            assert cache_state is not None
            final_state = cache_state
        else:
            state_plus = np.asarray(estimated_state_at_burn, dtype=float).copy()
            state_plus[3:] += dv1_mps / 1000.0
            final_state = propagate_state(
                propagator, state_plus, burn_jdtdb, acquisition_jdtdb
            )
            cache_x = dv1_mps.copy()
            cache_state = final_state
        return (final_state[:3] - np.asarray(target_position_km, dtype=float)) / scale

    result = least_squares(
        residual,
        np.asarray(initial_guess_mps, dtype=float),
        bounds=(
            np.full(3, -component_bound),
            np.full(3, component_bound),
        ),
        max_nfev=int(optimization_cfg.get("max_nfev", 300)),
        xtol=float(optimization_cfg.get("xtol", 1.0e-10)),
        ftol=float(optimization_cfg.get("ftol", 1.0e-10)),
        gtol=float(optimization_cfg.get("gtol", 1.0e-10)),
        verbose=0,
    )
    # Force one final evaluation so the returned state corresponds exactly to result.x.
    residual(result.x)
    assert cache_state is not None
    return np.asarray(result.x, dtype=float), result, cache_state.copy()


def optimize_dispersion_case(
    case: pd.Series,
    nominal_row: pd.Series,
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    config: dict[str, Any],
) -> dict[str, Any]:
    case_start = time.time()
    deadline = make_case_deadline(config, case_start)
    constructed = build_dispersed_initial_state(
        case, nominal_row, reference, propagator, config
    )
    target_state = np.array(
        [
            nominal_row["target_x_geo_eme_km"],
            nominal_row["target_y_geo_eme_km"],
            nominal_row["target_z_geo_eme_km"],
            nominal_row["target_vx_geo_eme_kms"],
            nominal_row["target_vy_geo_eme_kms"],
            nominal_row["target_vz_geo_eme_kms"],
        ],
        dtype=float,
    )
    acquisition_jdtdb = float(nominal_row["acquisition_jdtdb"])
    scheduled_burn = float(constructed["scheduled_first_burn_jdtdb"])
    truth_state_at_scheduled_burn = propagate_state(
        propagator,
        constructed["initial_state"],
        constructed["actual_deployment_jdtdb"],
        scheduled_burn,
    )

    navigation_error = np.hstack(
        [
            row_vector(
                case,
                [
                    "first_burn_navigation_position_error_x_km",
                    "first_burn_navigation_position_error_y_km",
                    "first_burn_navigation_position_error_z_km",
                ],
            ),
            row_vector(
                case,
                [
                    "first_burn_navigation_velocity_error_x_mps",
                    "first_burn_navigation_velocity_error_y_mps",
                    "first_burn_navigation_velocity_error_z_mps",
                ],
                scale=1.0 / 1000.0,
            ),
        ]
    )
    estimated_state_at_burn = truth_state_at_scheduled_burn + navigation_error
    nominal_dv1_guess = np.array(
        [nominal_row["dv1_x_mps"], nominal_row["dv1_y_mps"], nominal_row["dv1_z_mps"]],
        dtype=float,
    )
    dv1_command, optimizer_result, estimated_acquisition_minus = optimize_dv1_for_position(
        propagator,
        estimated_state_at_burn,
        scheduled_burn,
        acquisition_jdtdb,
        target_state[:3],
        nominal_dv1_guess,
        config,
        deadline,
    )
    dv2_command_from_estimate = 1000.0 * (
        target_state[3:] - estimated_acquisition_minus[3:]
    )

    # First burn execution may occur at a dispersed epoch and with vector error.
    actual_first_burn_jdtdb = scheduled_burn + float(
        case["first_burn_timing_error_s"]
    ) / 86400.0
    if actual_first_burn_jdtdb >= acquisition_jdtdb:
        raise TemporalInfeasibilityError(
            "Dispersed first-burn execution epoch is not before acquisition."
        )
    truth_state_at_actual_burn = propagate_state(
        propagator,
        constructed["initial_state"],
        constructed["actual_deployment_jdtdb"],
        actual_first_burn_jdtdb,
    )
    dv1_executed = perturb_burn_vector(
        dv1_command,
        float(case["first_burn_magnitude_fraction_error"]),
        float(case["first_burn_pointing_error_1_deg"]),
        float(case["first_burn_pointing_error_2_deg"]),
    )
    truth_state_after_first = truth_state_at_actual_burn.copy()
    truth_state_after_first[3:] += dv1_executed / 1000.0
    truth_acquisition_minus = propagate_state(
        propagator,
        truth_state_after_first,
        actual_first_burn_jdtdb,
        acquisition_jdtdb,
    )

    acquisition_navigation_error = np.hstack(
        [
            row_vector(
                case,
                [
                    "acquisition_navigation_position_error_x_km",
                    "acquisition_navigation_position_error_y_km",
                    "acquisition_navigation_position_error_z_km",
                ],
            ),
            row_vector(
                case,
                [
                    "acquisition_navigation_velocity_error_x_mps",
                    "acquisition_navigation_velocity_error_y_mps",
                    "acquisition_navigation_velocity_error_z_mps",
                ],
                scale=1.0 / 1000.0,
            ),
        ]
    )
    estimated_truth_acquisition_minus = truth_acquisition_minus + acquisition_navigation_error
    acquisition_navigation_enabled = as_bool(
        config["dispersion"].get("navigation_at_acquisition", {}).get("enabled", False)
    )
    if acquisition_navigation_enabled:
        dv2_command = 1000.0 * (
            target_state[3:] - estimated_truth_acquisition_minus[3:]
        )
    else:
        # In ideal Level-1 operation the acquisition burn is recomputed from the
        # actual pre-burn state. This equals the nominal derived burn when no
        # upstream navigation/execution errors are enabled.
        dv2_command = 1000.0 * (target_state[3:] - truth_acquisition_minus[3:])

    acquisition_timing_error_s = float(case["acquisition_burn_timing_error_s"])
    evaluation_jdtdb = acquisition_jdtdb
    if abs(acquisition_timing_error_s) > 0.0:
        # The target epoch stays fixed. A nonzero acquisition timing error is
        # represented by propagating the pre-burn state to the actual burn epoch,
        # applying the burn, then propagating back/forward to a configurable
        # verification epoch. To keep the terminal phase definition unambiguous,
        # the default config leaves this toggle disabled.
        actual_acquisition_burn = acquisition_jdtdb + acquisition_timing_error_s / 86400.0
        truth_state_at_acq_burn = propagate_state(
            propagator,
            truth_acquisition_minus,
            acquisition_jdtdb,
            actual_acquisition_burn,
        )
        evaluation_jdtdb = actual_acquisition_burn
    else:
        actual_acquisition_burn = acquisition_jdtdb
        truth_state_at_acq_burn = truth_acquisition_minus.copy()

    dv2_executed = perturb_burn_vector(
        dv2_command,
        float(case["acquisition_burn_magnitude_fraction_error"]),
        float(case["acquisition_burn_pointing_error_1_deg"]),
        float(case["acquisition_burn_pointing_error_2_deg"]),
    )
    final_state = truth_state_at_acq_burn.copy()
    final_state[3:] += dv2_executed / 1000.0

    # If the acquisition burn timing is dispersed, compare at the actual burn
    # epoch against the phase-zero target transformed to that epoch.
    evaluation_target = (
        target_state
        if np.isclose(evaluation_jdtdb, acquisition_jdtdb)
        else state_secr_to_eme(reference.phase_zero_state_secr, evaluation_jdtdb)
    )
    position_error = final_state[:3] - evaluation_target[:3]
    velocity_error_mps = 1000.0 * (final_state[3:] - evaluation_target[3:])
    position_error_norm = float(np.linalg.norm(position_error))
    velocity_error_norm = float(np.linalg.norm(velocity_error_mps))

    optimization_cfg = config["dispersion_optimization"]
    target_success = bool(
        position_error_norm
        <= float(optimization_cfg.get("position_success_tolerance_km", 1.0))
        and velocity_error_norm
        <= float(optimization_cfg.get("velocity_success_tolerance_mps", 1.0e-3))
    )

    record = dict(case)
    record.update(
        {
            "case_id": f"{nominal_row['separation_id']}_{case['dispersion_case_id']}",
            "source_nominal_separation_id": str(nominal_row["separation_id"]),
            "source_nominal_phase_rad": float(nominal_row["separation_phase_rad"]),
            "source_nominal_phase_deg": float(nominal_row["separation_phase_deg"]),
            "source_nominal_azimuth_deg": float(
                nominal_row["optimized_separation_azimuth_deg"]
            ),
            "source_nominal_elevation_deg": float(
                nominal_row["optimized_separation_elevation_deg"]
            ),
            "source_nominal_wait_days": float(
                nominal_row["wait_after_commissioning_days"]
            ),
            "deployment_jdtdb": float(nominal_row["deployment_jdtdb"]),
            "deployment_utc": str(nominal_row["deployment_utc"]),
            "actual_deployment_jdtdb": constructed["actual_deployment_jdtdb"],
            "actual_deployment_utc": constructed["actual_deployment_utc"],
            "actual_commissioning_duration_days": constructed[
                "actual_commissioning_duration_days"
            ],
            "scheduled_first_burn_jdtdb": scheduled_burn,
            "scheduled_first_burn_utc": jdtdb_to_utc(scheduled_burn),
            "actual_first_burn_jdtdb": actual_first_burn_jdtdb,
            "actual_first_burn_utc": jdtdb_to_utc(actual_first_burn_jdtdb),
            "remaining_transfer_duration_days": float(
                acquisition_jdtdb - actual_first_burn_jdtdb
            ),
            "acquisition_jdtdb": acquisition_jdtdb,
            "acquisition_utc": jdtdb_to_utc(acquisition_jdtdb),
            "actual_separation_speed_mps": constructed["actual_separation_speed_mps"],
            "actual_separation_azimuth_rad": constructed[
                "actual_separation_azimuth_rad"
            ],
            "actual_separation_azimuth_deg": float(
                np.rad2deg(constructed["actual_separation_azimuth_rad"])
            ),
            "actual_separation_elevation_rad": constructed[
                "actual_separation_elevation_rad"
            ],
            "actual_separation_elevation_deg": float(
                np.rad2deg(constructed["actual_separation_elevation_rad"])
            ),
            "dv1_x_mps": float(dv1_command[0]),
            "dv1_y_mps": float(dv1_command[1]),
            "dv1_z_mps": float(dv1_command[2]),
            "dv1_mag_mps": float(np.linalg.norm(dv1_command)),
            "dv1_executed_x_mps": float(dv1_executed[0]),
            "dv1_executed_y_mps": float(dv1_executed[1]),
            "dv1_executed_z_mps": float(dv1_executed[2]),
            "dv1_executed_mag_mps": float(np.linalg.norm(dv1_executed)),
            "dv2_x_mps": float(dv2_command[0]),
            "dv2_y_mps": float(dv2_command[1]),
            "dv2_z_mps": float(dv2_command[2]),
            "dv2_mag_mps": float(np.linalg.norm(dv2_command)),
            "dv2_executed_x_mps": float(dv2_executed[0]),
            "dv2_executed_y_mps": float(dv2_executed[1]),
            "dv2_executed_z_mps": float(dv2_executed[2]),
            "dv2_executed_mag_mps": float(np.linalg.norm(dv2_executed)),
            "dv_total_mps": float(np.linalg.norm(dv1_command) + np.linalg.norm(dv2_command)),
            "dv_total_executed_mps": float(
                np.linalg.norm(dv1_executed) + np.linalg.norm(dv2_executed)
            ),
            "optimizer_success": bool(optimizer_result.success),
            "target_success": target_success,
            "optimization_status": int(optimizer_result.status),
            "optimization_message": str(optimizer_result.message),
            "optimization_nfev": int(optimizer_result.nfev),
            "optimization_timed_out": False,
            "optimization_runtime_seconds": float(time.time() - case_start),
            "final_position_error_x_km": float(position_error[0]),
            "final_position_error_y_km": float(position_error[1]),
            "final_position_error_z_km": float(position_error[2]),
            "final_position_error_norm_km": position_error_norm,
            "final_velocity_error_x_mps": float(velocity_error_mps[0]),
            "final_velocity_error_y_mps": float(velocity_error_mps[1]),
            "final_velocity_error_z_mps": float(velocity_error_mps[2]),
            "final_velocity_error_norm_mps": velocity_error_norm,
            "trajectory_saved": False,
            "trajectory_file": "",
            "maneuver_file": "",
        }
    )

    for prefix, state in [
        ("initial", constructed["initial_state"]),
        ("target", evaluation_target),
        ("final", final_state),
    ]:
        record.update(
            {
                f"{prefix}_x_geo_eme_km": float(state[0]),
                f"{prefix}_y_geo_eme_km": float(state[1]),
                f"{prefix}_z_geo_eme_km": float(state[2]),
                f"{prefix}_vx_geo_eme_kms": float(state[3]),
                f"{prefix}_vy_geo_eme_kms": float(state[4]),
                f"{prefix}_vz_geo_eme_kms": float(state[5]),
            }
        )
    return record


def failed_dispersion_record(case: pd.Series, exception: Exception) -> dict[str, Any]:
    record = dict(case)
    timed_out = isinstance(exception, OptimizationTimeoutError)
    record.update(
        {
            "case_id": str(case.get("dispersion_case_id", "")),
            "optimizer_success": False,
            "target_success": False,
            "optimization_status": -998 if timed_out else -999,
            "optimization_message": str(exception),
            "optimization_timed_out": timed_out,
            "dv_total_mps": np.nan,
            "dv_total_executed_mps": np.nan,
            "final_position_error_norm_km": np.nan,
            "final_velocity_error_norm_mps": np.nan,
            "trajectory_saved": False,
            "trajectory_file": "",
            "maneuver_file": "",
        }
    )
    return record


# =============================================================================
# Dispersion ranking, statistics, representative outputs, and plots
# =============================================================================


def assign_empirical_percentiles(
    dataframe: pd.DataFrame,
    metric: str,
    success_column: str = "target_success",
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    percentile_column = f"{metric}_percentile"
    dataframe[percentile_column] = np.nan
    valid = dataframe[(dataframe[success_column] == True) & dataframe[metric].notna()].copy()
    if valid.empty:
        return dataframe
    ranks = valid[metric].rank(method="average", ascending=True)
    count = len(valid)
    percentiles = np.zeros(count) if count == 1 else 100.0 * (ranks - 1.0) / (count - 1.0)
    dataframe.loc[valid.index, percentile_column] = percentiles.to_numpy(dtype=float)
    return dataframe


def dispersion_statistics(
    dataframe: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    valid = dataframe[(dataframe["target_success"] == True) & dataframe[metric].notna()]
    record: dict[str, Any] = {
        "n_requested_cases": int(len(dataframe)),
        "n_optimizer_success": int((dataframe["optimizer_success"] == True).sum()),
        "n_target_success": int((dataframe["target_success"] == True).sum()),
        "n_failed": int((dataframe["target_success"] != True).sum()),
        "n_timed_out": int((dataframe.get("optimization_timed_out", False) == True).sum()),
        "success_rate": float((dataframe["target_success"] == True).mean()) if len(dataframe) else 0.0,
        "percentile_metric": metric,
    }
    if valid.empty:
        return pd.DataFrame([record])
    values = valid[metric].to_numpy(dtype=float)
    record.update(
        {
            f"{metric}_min": float(np.min(values)),
            f"{metric}_mean": float(np.mean(values)),
            f"{metric}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            f"{metric}_p05": float(np.percentile(values, 5)),
            f"{metric}_p25": float(np.percentile(values, 25)),
            f"{metric}_p50": float(np.percentile(values, 50)),
            f"{metric}_p75": float(np.percentile(values, 75)),
            f"{metric}_p90": float(np.percentile(values, 90)),
            f"{metric}_p95": float(np.percentile(values, 95)),
            f"{metric}_p99": float(np.percentile(values, 99)),
            f"{metric}_max": float(np.max(values)),
            "dv1_p50_mps": float(np.percentile(valid["dv1_mag_mps"], 50)),
            "dv1_p95_mps": float(np.percentile(valid["dv1_mag_mps"], 95)),
            "dv2_p50_mps": float(np.percentile(valid["dv2_mag_mps"], 50)),
            "dv2_p95_mps": float(np.percentile(valid["dv2_mag_mps"], 95)),
            "position_error_p95_km": float(
                np.percentile(valid["final_position_error_norm_km"], 95)
            ),
            "velocity_error_p95_mps": float(
                np.percentile(valid["final_velocity_error_norm_mps"], 95)
            ),
        }
    )
    return pd.DataFrame([record])


def select_representative_indices(
    dataframe: pd.DataFrame,
    metric: str,
    requested_percentiles: list[float],
) -> dict[int, list[float]]:
    valid = dataframe[(dataframe["target_success"] == True) & dataframe[metric].notna()]
    if valid.empty:
        return {}
    values = valid[metric].to_numpy(dtype=float)
    mapping: dict[int, list[float]] = {}
    for percentile in requested_percentiles:
        target_value = float(np.percentile(values, float(percentile)))
        local_index = int(np.argmin(np.abs(values - target_value)))
        dataframe_index = int(valid.index[local_index])
        mapping.setdefault(dataframe_index, []).append(float(percentile))
    return mapping


def save_dispersion_representative(
    dataframe: pd.DataFrame,
    dataframe_index: int,
    labels: list[float],
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[str, str]:
    row = dataframe.loc[dataframe_index]
    label_text = "_".join(f"p{int(round(value)):02d}" for value in labels)
    filename_base = f"{label_text}_{safe_filename(str(row['dispersion_case_id']))}"
    representative_dir = output_dir / "representatives"
    representative_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = representative_dir / f"{filename_base}_trajectory.csv"
    maneuver_path = representative_dir / f"{filename_base}_maneuvers.csv"

    state0 = np.array(
        [
            row["initial_x_geo_eme_km"],
            row["initial_y_geo_eme_km"],
            row["initial_z_geo_eme_km"],
            row["initial_vx_geo_eme_kms"],
            row["initial_vy_geo_eme_kms"],
            row["initial_vz_geo_eme_kms"],
        ],
        dtype=float,
    )
    dv1 = np.array([row["dv1_executed_x_mps"], row["dv1_executed_y_mps"], row["dv1_executed_z_mps"]])
    dv2 = np.array([row["dv2_executed_x_mps"], row["dv2_executed_y_mps"], row["dv2_executed_z_mps"]])
    step_days = float(config.get("output", {}).get("trajectories", {}).get("step_days", 0.05))
    times, microsat_states, events = propagate_impulsive_trajectory(
        propagator,
        state0,
        float(row["actual_deployment_jdtdb"]),
        float(row["actual_first_burn_jdtdb"]),
        float(row["acquisition_jdtdb"]),
        dv1,
        dv2,
        step_days,
    )
    mothership_states = mothership_reference_states(propagator, reference, times)
    phase_reference_states = phase_zero_reference_states(reference, times)
    bundle = trajectory_bundle_dataframe(
        times,
        microsat_states,
        mothership_states,
        phase_reference_states,
        events,
        reference.start_jdtdb,
    )
    bundle.to_csv(trajectory_path, index=False)

    maneuver_row = row.copy()
    maneuver_row["deployment_jdtdb"] = row["actual_deployment_jdtdb"]
    maneuver_row["first_burn_jdtdb"] = row["actual_first_burn_jdtdb"]
    maneuver_row["dv1_x_mps"] = row["dv1_executed_x_mps"]
    maneuver_row["dv1_y_mps"] = row["dv1_executed_y_mps"]
    maneuver_row["dv1_z_mps"] = row["dv1_executed_z_mps"]
    maneuver_row["dv2_x_mps"] = row["dv2_executed_x_mps"]
    maneuver_row["dv2_y_mps"] = row["dv2_executed_y_mps"]
    maneuver_row["dv2_z_mps"] = row["dv2_executed_z_mps"]
    save_maneuver_file(maneuver_path, maneuver_row, propagator)
    return str(trajectory_path), str(maneuver_path)


def save_dispersion_plots(
    dataframe: pd.DataFrame,
    metric: str,
    plot_dir: Path,
    config: dict[str, Any],
) -> None:
    if not as_bool(config.get("plots", {}).get("enabled", True)):
        return
    valid = dataframe[(dataframe["target_success"] == True) & dataframe[metric].notna()]
    if valid.empty:
        return
    size, _, extension = plot_configuration(config)
    plot_dir.mkdir(parents=True, exist_ok=True)
    values = valid[metric].to_numpy(dtype=float)

    if as_bool(config.get("plots", {}).get("save_dispersion_histogram", True)):
        fig, ax = plt.subplots(figsize=size)
        ax.hist(values, bins="auto")
        ax.set_xlabel(f"{metric} [m/s]")
        ax.set_ylabel("Count")
        ax.set_title("Deployment dispersion distribution")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"dispersion_histogram.{extension}", bbox_inches="tight")
        plt.close(fig)

    if as_bool(config.get("plots", {}).get("save_dispersion_cdf", True)):
        sorted_values = np.sort(values)
        probabilities = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
        fig, ax = plt.subplots(figsize=size)
        ax.plot(sorted_values, probabilities)
        ax.set_xlabel(f"{metric} [m/s]")
        ax.set_ylabel("Empirical cumulative probability")
        ax.set_title("Deployment dispersion CDF")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"dispersion_cdf.{extension}", bbox_inches="tight")
        plt.close(fig)

    if as_bool(config.get("plots", {}).get("save_burn_component_distributions", True)):
        for column, filename, xlabel in [
            ("dv1_mag_mps", "dv1_histogram", "First correction $\\Delta V$ [m/s]"),
            ("dv2_mag_mps", "dv2_histogram", "Acquisition $\\Delta V$ [m/s]"),
        ]:
            fig, ax = plt.subplots(figsize=size)
            ax.hist(valid[column].to_numpy(dtype=float), bins="auto")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{filename}.{extension}", bbox_inches="tight")
            plt.close(fig)


# =============================================================================
# Run modes
# =============================================================================


def write_common_metadata(
    output_dir: Path,
    config: dict[str, Any],
    reference: ReferenceOrbit,
    run_mode: str,
    run_start_time: float,
    extra: dict[str, Any],
) -> None:
    metadata_cfg = config.get("output", {}).get("metadata", {})
    metadata = {
        "run_mode": run_mode,
        "elapsed_wallclock_seconds": float(time.time() - run_start_time),
        "elapsed_wallclock_hhmmss": format_elapsed(time.time() - run_start_time),
        "reference_csv": str(config["files"]["lpf_orbit_csv"]),
        "reference_csv_sha256": sha256_file(Path(config["files"]["lpf_orbit_csv"])),
        "reference_period_start_index": reference.start_index,
        "reference_period_end_index_inclusive": reference.end_index,
        "reference_period_start_jdtdb": reference.start_jdtdb,
        "reference_period_start_utc": jdtdb_to_utc(reference.start_jdtdb),
        "reference_period_end_jdtdb": reference.end_jdtdb,
        "reference_period_end_utc": jdtdb_to_utc(reference.end_jdtdb),
        "reference_period_days": reference.period_days,
        "phase_convention": "Deployment and microsatellite target phase are 0 rad; mothership-ahead separation is positive and does not wrap beyond one configured period.",
        "state_frame": "Earth-centred EME/J2000",
        "separation_direction_frame": config["separation"].get(
            "direction_frame", "local_quasi_halo"
        ),
        "propagation": config["propagation"],
        "spice_kernels": config["spice"]["kernels"],
        **extra,
    }
    if as_bool(metadata_cfg.get("save_json", True)):
        save_json(
            output_dir / metadata_cfg.get("filename", "run_metadata.json"),
            metadata,
        )
    if as_bool(metadata_cfg.get("save_resolved_config", True)):
        save_yaml(
            output_dir
            / metadata_cfg.get("resolved_config_filename", "resolved_config.yaml"),
            config,
        )


def run_nominal_sweep(
    config: dict[str, Any],
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    comm: MPI.Comm,
    run_start_time: float,
    output_dir: Path,
) -> None:
    rank = comm.Get_rank()
    phase_contexts = build_phase_contexts(reference, config)
    if rank == 0:
        coarse_tasks = build_coarse_wait_tasks(phase_contexts, config)
        print(
            f"Nominal sweep: {len(phase_contexts)} phase separations, "
            f"{len(coarse_tasks)} coarse phase-wait tasks.",
            flush=True,
        )
    else:
        coarse_tasks = None
    coarse_tasks = comm.bcast(coarse_tasks, root=0)

    coarse_results, stopped_coarse, assigned_coarse, completed_coarse = run_nominal_task_stage(
        coarse_tasks,
        "coarse",
        phase_contexts,
        reference,
        propagator,
        config,
        comm,
        run_start_time,
        output_dir,
    )

    if rank == 0:
        assert coarse_results is not None
        refinement_tasks = build_refinement_wait_tasks(
            coarse_results, phase_contexts, config
        )
    else:
        refinement_tasks = None
    refinement_tasks = comm.bcast(refinement_tasks, root=0)

    if refinement_tasks is not None and not refinement_tasks.empty:
        refinement_results, stopped_refine, assigned_refine, completed_refine = run_nominal_task_stage(
            refinement_tasks,
            "refine",
            phase_contexts,
            reference,
            propagator,
            config,
            comm,
            run_start_time,
            output_dir,
        )
    else:
        refinement_results = pd.DataFrame() if rank == 0 else None
        stopped_refine = []
        assigned_refine = []
        completed_refine = []

    if rank != 0:
        return
    assert coarse_results is not None
    assert refinement_results is not None
    all_wait_results = pd.concat(
        [coarse_results, refinement_results], ignore_index=True, sort=False
    )
    all_wait_results = all_wait_results.sort_values(
        ["separation_phase_rad", "wait_after_commissioning_days", "stage"]
    ).reset_index(drop=True)
    nominal_master = select_nominal_phase_results(all_wait_results)

    nominal_cfg = config.get("output", {}).get("nominal", {})
    master_path = output_dir / nominal_cfg.get(
        "master_filename", "nominal_separation_sweep_summary.csv"
    )
    wait_master_path = output_dir / nominal_cfg.get(
        "wait_master_filename", "nominal_wait_sweep_master.csv"
    )
    all_wait_results.to_csv(wait_master_path, index=False)

    apply_plot_font(config)
    trajectory_files: list[str] = []
    maneuver_files: list[str] = []
    wait_files: list[str] = []
    for _, row in nominal_master.iterrows():
        if as_bool(nominal_cfg.get("save_all_trajectory_bundles", True)):
            trajectory_file, maneuver_file, wait_file = save_nominal_trajectory_and_files(
                row,
                all_wait_results,
                reference,
                propagator,
                output_dir,
                config,
            )
        else:
            trajectory_file = maneuver_file = wait_file = ""
        trajectory_files.append(trajectory_file)
        maneuver_files.append(maneuver_file)
        wait_files.append(wait_file)
    if len(nominal_master):
        nominal_master["trajectory_file"] = trajectory_files
        nominal_master["maneuver_file"] = maneuver_files
        nominal_master["wait_sweep_file"] = wait_files
    nominal_master.to_csv(master_path, index=False)

    save_nominal_summary_plots(nominal_master, output_dir / "plots", config)

    stopped_flags = list(stopped_coarse) + list(stopped_refine)
    write_common_metadata(
        output_dir,
        config,
        reference,
        "nominal_sweep",
        run_start_time,
        {
            "partial_run": bool(any(stopped_flags)),
            "coarse_assigned_counts": assigned_coarse,
            "coarse_completed_counts": completed_coarse,
            "refine_assigned_counts": assigned_refine,
            "refine_completed_counts": completed_refine,
            "n_phase_separations": len(phase_contexts),
            "n_wait_results": len(all_wait_results),
            "n_nominal_target_success": int(
                (nominal_master.get("target_success", False) == True).sum()
            ),
            "nominal_master_csv": str(master_path),
            "nominal_wait_master_csv": str(wait_master_path),
        },
    )

    print("Nominal deployment sweep complete")
    print(f"Master CSV: {master_path}")
    print(f"Wait master: {wait_master_path}")
    print(f"Successful nominal phases: {(nominal_master['target_success'] == True).sum()}/{len(nominal_master)}")


def load_source_nominal_row(config: dict[str, Any]) -> pd.Series:
    source_cfg = config["dispersion_source_nominal"]
    path = Path(source_cfg["nominal_master_csv"])
    dataframe = pd.read_csv(path)
    require_success = as_bool(source_cfg.get("require_target_success", True))
    if require_success and "target_success" in dataframe.columns:
        dataframe = dataframe[dataframe["target_success"] == True].copy()
    selection_mode = str(source_cfg.get("selection_mode", "separation_id")).lower()
    if selection_mode == "separation_id":
        separation_id = str(source_cfg["separation_id"])
        subset = dataframe[dataframe["separation_id"].astype(str) == separation_id]
    elif selection_mode == "phase_deg":
        target = float(source_cfg["phase_deg"])
        tolerance = float(source_cfg.get("phase_tolerance_deg", 1.0e-6))
        subset = dataframe[
            np.abs(dataframe["separation_phase_deg"].to_numpy(dtype=float) - target)
            <= tolerance
        ]
    elif selection_mode == "row_index":
        index = int(source_cfg["row_index"])
        if index < 0 or index >= len(dataframe):
            raise IndexError("dispersion_source_nominal.row_index is outside nominal master.")
        return dataframe.iloc[index]
    else:
        raise ValueError(f"Unknown dispersion source selection mode: {selection_mode}")
    if subset.empty:
        raise RuntimeError("No nominal row matches the dispersion source selection.")
    if len(subset) > 1:
        raise RuntimeError("Dispersion source selection matched more than one nominal row.")
    return subset.iloc[0]


def run_dispersion(
    config: dict[str, Any],
    reference: ReferenceOrbit,
    propagator: NBodyPropagator,
    comm: MPI.Comm,
    run_start_time: float,
    output_dir: Path,
) -> None:
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        nominal_row = load_source_nominal_row(config)
        cases = generate_dispersion_cases(nominal_row, reference, config)
        print(
            f"Dispersion run for {nominal_row['separation_id']}: "
            f"{len(cases)} cases.",
            flush=True,
        )
    else:
        nominal_row = None
        cases = None
    nominal_row, cases = comm.bcast((nominal_row, cases), root=0)
    assert nominal_row is not None
    assert cases is not None

    local_cases = cases.iloc[rank::size].copy()
    local_records: list[dict[str, Any]] = []
    stopped = False
    progress_every = int(config.get("mpi", {}).get("progress_every", 1))
    for local_index, (_, case) in enumerate(local_cases.iterrows()):
        if should_save_and_stop(config, run_start_time):
            stopped = True
            break
        if progress_every > 0 and (
            local_index == 0 or (local_index + 1) % progress_every == 0
        ):
            print(
                f"[rank {rank:04d}/{size:04d}] dispersion case "
                f"{local_index + 1}/{len(local_cases)}: {case['dispersion_case_id']}",
                flush=True,
            )
        try:
            record = optimize_dispersion_case(
                case, nominal_row, reference, propagator, config
            )
        except Exception as exception:
            record = failed_dispersion_record(case, exception)
        local_records.append(record)

    save_rank_records(
        output_dir,
        mode="dispersion",
        stage="cases",
        rank=rank,
        records=local_records,
        enabled=as_bool(config.get("mpi", {}).get("save_rank_outputs", True)),
    )
    stopped_flags = comm.gather(stopped, root=0)
    assigned_counts = comm.gather(len(local_cases), root=0)
    completed_counts = comm.gather(len(local_records), root=0)
    gathered = comm.gather(local_records, root=0)
    if rank != 0:
        return

    records = [record for block in gathered for record in block]
    master = pd.DataFrame(records)
    if not master.empty:
        master = master.sort_values("sample_id").reset_index(drop=True)
    output_cfg = config.get("output", {}).get("dispersion", {})
    metric = str(output_cfg.get("percentile_metric", "dv_total_mps"))
    master = assign_empirical_percentiles(master, metric)

    representative_percentiles = [
        float(value)
        for value in output_cfg.get(
            "representative_percentiles", [0, 50, 95, 99, 100]
        )
    ]
    representative_mapping = select_representative_indices(
        master, metric, representative_percentiles
    )
    representative_info: list[dict[str, Any]] = []
    if as_bool(output_cfg.get("save_representative_trajectories", True)):
        for dataframe_index, labels in representative_mapping.items():
            trajectory_file, maneuver_file = save_dispersion_representative(
                master,
                dataframe_index,
                labels,
                reference,
                propagator,
                output_dir,
                config,
            )
            label_text = ";".join(f"p{int(round(value)):02d}" for value in labels)
            master.loc[dataframe_index, "representative_labels"] = label_text
            master.loc[dataframe_index, "trajectory_saved"] = True
            master.loc[dataframe_index, "trajectory_file"] = trajectory_file
            master.loc[dataframe_index, "maneuver_file"] = maneuver_file
            representative_info.append(
                {
                    "case_id": master.loc[dataframe_index, "case_id"],
                    "labels": labels,
                    "metric_value": master.loc[dataframe_index, metric],
                    "trajectory_file": trajectory_file,
                    "maneuver_file": maneuver_file,
                }
            )

    master_filename = output_cfg.get(
        "master_filename", "deployment_dispersion_master.csv"
    )
    statistics_filename = output_cfg.get(
        "statistics_filename", "dispersion_statistics.csv"
    )
    master_path = output_dir / master_filename
    statistics_path = output_dir / statistics_filename
    master.to_csv(master_path, index=False)
    stats = dispersion_statistics(master, metric)
    stats.insert(0, "separation_id", str(nominal_row["separation_id"]))
    stats.insert(1, "separation_phase_rad", float(nominal_row["separation_phase_rad"]))
    stats.insert(2, "separation_phase_deg", float(nominal_row["separation_phase_deg"]))
    stats.to_csv(statistics_path, index=False)

    source_nominal_path = output_dir / "source_nominal_solution.json"
    save_json(source_nominal_path, nominal_row.to_dict())
    apply_plot_font(config)
    save_dispersion_plots(master, metric, output_dir / "plots", config)

    write_common_metadata(
        output_dir,
        config,
        reference,
        "dispersion",
        run_start_time,
        {
            "partial_run": bool(any(stopped_flags)),
            "source_nominal_master_csv": config["dispersion_source_nominal"][
                "nominal_master_csv"
            ],
            "source_nominal_separation_id": str(nominal_row["separation_id"]),
            "source_nominal_phase_rad": float(nominal_row["separation_phase_rad"]),
            "source_nominal_phase_deg": float(nominal_row["separation_phase_deg"]),
            "assigned_counts": assigned_counts,
            "completed_counts": completed_counts,
            "n_requested_cases": len(cases),
            "n_completed_cases": len(master),
            "n_target_success": int((master.get("target_success", False) == True).sum()),
            "n_timed_out": int((master.get("optimization_timed_out", False) == True).sum()),
            "percentile_metric": metric,
            "representatives": representative_info,
            "master_csv": str(master_path),
            "statistics_csv": str(statistics_path),
        },
    )

    print("Deployment dispersion run complete")
    print(f"Master CSV: {master_path}")
    print(f"Statistics CSV: {statistics_path}")
    print(f"Target successes: {(master['target_success'] == True).sum()}/{len(master)}")


# =============================================================================
# CLI and main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPI quasi-halo microsatellite deployment optimization and dispersion."
    )
    parser.add_argument("config", help="Path to the YAML configuration file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    run_start_time = time.time()

    run_mode = str(config.get("run", {}).get("mode", "nominal_sweep")).lower()
    output_dir = Path(config["output"]["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    load_spice_kernels(config)
    reference = load_reference_orbit(config)
    propagator = build_nbody_propagator(config)

    if rank == 0:
        print("Quasi-halo microsatellite deployment analysis")
        print("---------------------------------------------")
        print(f"Run mode:          {run_mode}")
        print(f"MPI ranks:         {size}")
        print(f"Reference period:  {reference.period_days:.9f} days")
        print(f"Deployment UTC:    {jdtdb_to_utc(reference.start_jdtdb)}")
        print(f"Output directory:  {output_dir}")
        print()

    if run_mode == "nominal_sweep":
        run_nominal_sweep(
            config,
            reference,
            propagator,
            comm,
            run_start_time,
            output_dir,
        )
    elif run_mode == "dispersion":
        run_dispersion(
            config,
            reference,
            propagator,
            comm,
            run_start_time,
            output_dir,
        )
    else:
        raise ValueError("run.mode must be 'nominal_sweep' or 'dispersion'.")


if __name__ == "__main__":
    main()
