from __future__ import annotations

"""Standalone study of initial boresight packing near the Earth--Moon IV zone.

The script compares two concepts using the LPF orbit named by the mission YAML:

1. ``fixed_global``
   One common SECR tangent and one set of fixed spacecraft boresights are
   selected.  The complete FOV of every spacecraft must clear its dynamic
   Earth--Moon invisibility zone at every sampled epoch.

2. ``dynamic_per_spacecraft``
   At every sampled epoch, every spacecraft receives its own tangent from its
   own instantaneous Earth--Moon IV-zone axis.  The nominal boresight is kept
   whenever its complete FOV is feasible.  If it is not, that spacecraft is
   moved outward along its own tangent by the minimum amount required for
   clearance.  Additional spacecraft are then packed inward from the nominal
   direction while feasible; after the first inward failure, all remaining
   spacecraft are packed outward.  Actual pairwise angular separations are
   checked because the spacecraft tangents are not identical.

The FOV spacing is calculated from the configured sky-area FOV using the same
spherical-cap equivalent circular cone used by Spacecraft.py.

For user-selected epochs, the script also writes 3-D SECR scenes containing
Earth, the Moon orbit, the full LPF quasi-halo orbit, the instantaneous
spacecraft locations, the selected-mode FOV cones, and optional IV cones.

Required neighboring files/data
--------------------------------
- Mission YAML passed on the command line.
- LPF orbit CSV referenced by ``orbit_file_path`` in that YAML.
- earth_moon_invisibility_zone.py in the Python path.

Example
-------
python study_dynamic_initial_boresight_packing.py \
    sc_2_overall_orbitdetsim_config.yaml

For a headless run:
python study_dynamic_initial_boresight_packing.py \
    sc_2_overall_orbitdetsim_config.yaml --no-show
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from earth_moon_invisibility_zone import (
    compute_earth_moon_invisibility_zone_batch,
)


# =============================================================================
# Study options -- edit these directly for now
# =============================================================================

# Existing SECR convention used by the current simulation.
NOMINAL_BORESIGHT_SECR = np.array([-1.0, 0.0, 0.0], dtype=float)

# Use config['num_spacecraft'] when None.
NUM_SPACECRAFT_OVERRIDE=4

# Relative halo phases in degrees.  None gives equal spacing.  The first value
# must be zero when a custom list is supplied.
RELATIVE_PHASE_DEG=[0.0, 5.0, 10.0, 15.0]

# Index of SC1 inside the one-period LPF slice.  This is deterministic for the
# study.  Change it to examine another common formation phase.
ANCHOR_PERIOD_INDEX = 0

# Dynamic Earth--Moon IV-zone definition.
IV_HALF_ANGLE_DEG = 12.0
IV_CLEARANCE_MARGIN_DEG = 0.25

# Additional centre-to-centre separation beyond exactly touching circular FOVs.
FOV_SEPARATION_MARGIN_DEG = 0.25


# Policy for epochs where the complete nominal FOV intersects that
# spacecraft's own IV zone:
#   'shift_outward' -> move minimally away from the IV axis along that
#                      spacecraft's instantaneous tangent.
#   'fail'          -> preserve the old strict behavior and raise an error.
NOMINAL_VIOLATION_POLICY = "shift_outward"

# Small numerical buffer added beyond the analytical IV boundary after an
# outward correction.  This is in addition to IV_CLEARANCE_MARGIN_DEG.
OUTWARD_CLEARANCE_BUFFER_DEG = 1.0e-4

# Angular increment used only when actual pairwise separation requires a
# spacecraft to move farther than its nominal +/-N*spacing slot.  The IV
# boundary correction itself is analytical, not discretized by this value.
PACKING_SEARCH_STEP_DEG = 0.01

# Maximum magnitude searched on the outward side of the nominal direction.
MAX_OUTWARD_OFFSET_DEG = 90.0

# Packing solutions to execute.  ``dynamic_per_spacecraft`` is the
# epochwise mode: every spacecraft gets a new tangent and boresight at every
# sampled LPF epoch from its own instantaneous IV-zone geometry.
MODES_TO_RUN = ['dynamic_per_spacecraft']

# Fixed-mode tangent source:
#   'initial' -> tangent from the first sampled epoch
#   'mean'    -> tangent from the normalized mean IV axis over all epochs/SCs
# This option affects only ``fixed_global``.  It has no effect on the
# epochwise ``dynamic_per_spacecraft`` solution.
FIXED_TANGENT_SOURCE = "initial"

# Sample every Nth LPF row for the boresight/IV design calculation.
# Use 1 for every orbit row; a larger value makes exploratory runs much faster.
DESIGN_SAMPLE_STRIDE = 250

# Epochs at which projected FOV/IV footprints are visualized.
SNAPSHOT_FRACTIONS = (0.0, 0.5)

# Epochs for the new 3-D orbit/FOV-cone scenes.  Choose one selection mode:
#   'sampled_index'       -> values index the downsampled design epochs
#   'source_period_index' -> values index rows inside the full one-period LPF
#                            slice; the nearest sampled design epoch is used
#   'fraction'            -> values lie in [0, 1] over the sampled period
# Examples:
#   THREE_D_EPOCH_SELECTION_MODE = 'sampled_index'
#   THREE_D_EPOCH_VALUES = (0, 10, 20)
# or
#   THREE_D_EPOCH_SELECTION_MODE = 'source_period_index'
#   THREE_D_EPOCH_VALUES = (0, 5000, 10000)
THREE_D_EPOCH_SELECTION_MODE = "source_period_index"
THREE_D_EPOCH_VALUES = (0,10_000, 50_000)

# Physical length used only to draw the FOV and IV cones in the 3-D scene.
# This does not affect any boresight or clearance calculation.
THREE_D_CONE_LENGTH_KM = 1_500_000.0
THREE_D_CONE_AZIMUTH_SAMPLES = 49
THREE_D_CONE_AXIAL_SAMPLES = 8
THREE_D_SHOW_IV_CONES = True
THREE_D_SHOW_EARTH_MOON_LOS = False
THREE_D_ORBIT_DECIMATION = 1

# Distances ahead of the formation along the nominal SECR boresight.  Because
# the nominal is x-aligned, these are x=constant planes and are displayed in
# the y-z plane.
CROSS_SECTION_DISTANCES_KM = (10_000.0, 250_000.0, 750_000.0)

# Number of angular samples used to draw each cone-plane intersection.
CONE_BOUNDARY_SAMPLES = 361

# Output folder, interpreted relative to the YAML location when not absolute.
OUTPUT_FOLDER = "initial_boresight_packing_study"

# Open figures interactively unless --no-show is passed.
SHOW_FIGURES_BY_DEFAULT = True


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class StudyGeometry:
    times: np.ndarray
    source_indices: np.ndarray
    spacecraft_positions_secr_km: np.ndarray  # (E, S, 3)
    moon_positions_secr_km: np.ndarray  # (E, 3)
    spacecraft_orbit_tracks_secr_km: np.ndarray  # (P, S, 3), full period
    moon_orbit_track_secr_km: np.ndarray  # (P, 3), full period
    iv_axes_secr: np.ndarray  # (E, S, 3)
    iv_half_angle_rad: float
    requested_phase_deg: np.ndarray
    realized_phase_deg: np.ndarray
    phase_offset_indices: np.ndarray


@dataclass(frozen=True)
class PackingResult:
    mode: str
    boresights_secr: np.ndarray  # (E, S, 3)
    offsets_rad: np.ndarray  # (E, S), signed from NOMINAL_BORESIGHT_SECR
    inward_count: np.ndarray  # (E,)
    iv_clearance_rad: np.ndarray  # (E, S)
    tangent_secr: np.ndarray  # (E, S, 3); fixed mode repeats one tangent
    nominal_clearance_rad: np.ndarray  # (E, S), before any correction
    minimum_outward_shift_rad: np.ndarray  # (E, S), non-positive


# =============================================================================
# Basic geometry
# =============================================================================


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-15:
        raise ValueError(f"{name} must be a finite nonzero vector.")
    return value / norm


def _normalize_rows(vectors: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(values, axis=-1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-15):
        bad = np.argwhere((~np.isfinite(norms)) | (norms <= 1.0e-15))
        raise ValueError(f"{name} contains invalid vectors at {bad.tolist()}.")
    return values / norms[..., None]


def fov_area_deg2_to_half_angle_rad(fov_deg2: float) -> float:
    """Equivalent circular-cone half-angle from sky area in square degrees."""

    area = float(fov_deg2)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError(f"FOV area must be finite and positive, got {fov_deg2!r}.")
    steradians = area / (180.0 / np.pi) ** 2
    argument = 1.0 - steradians / (2.0 * np.pi)
    if not -1.0 <= argument <= 1.0:
        raise ValueError(f"FOV area {area} deg^2 is too large for a spherical cap.")
    return float(np.arccos(argument))


def angular_separation_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_hat = _normalize_rows(np.asarray(a, dtype=float), "first direction")
    b_hat = _normalize_rows(np.asarray(b, dtype=float), "second direction")
    return np.arccos(np.clip(np.sum(a_hat * b_hat, axis=-1), -1.0, 1.0))


def tangent_toward_axis(nominal: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Project an IV axis into the tangent plane at the nominal boresight."""

    b0 = _unit(nominal, "nominal boresight")
    a = _unit(axis, "IV axis")
    tangent = a - np.dot(a, b0) * b0
    norm = float(np.linalg.norm(tangent))
    if norm <= 1.0e-12:
        # Deterministic fallback if nominal and IV axis are collinear.
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(trial, b0))) > 0.95:
            trial = np.array([0.0, 0.0, 1.0], dtype=float)
        tangent = trial - np.dot(trial, b0) * b0
    return _unit(tangent, "Earthward tangent")


def boresight_on_tangent(
    nominal: np.ndarray,
    tangent: np.ndarray,
    signed_offset_rad: float | np.ndarray,
) -> np.ndarray:
    """Rotate nominal along one signed great-circle tangent."""

    b0 = _unit(nominal, "nominal boresight")
    t = _unit(tangent, "tangent")
    if abs(float(np.dot(b0, t))) > 1.0e-10:
        raise ValueError("nominal and tangent must be perpendicular.")
    delta = np.asarray(signed_offset_rad, dtype=float)
    result = np.cos(delta)[..., None] * b0 + np.sin(delta)[..., None] * t
    return _normalize_rows(result, "constructed boresight")


def packed_offsets_rad(
    num_spacecraft: int,
    inward_count: int,
    separation_rad: float,
) -> np.ndarray:
    """SC1 at zero, then inward slots, then outward slots."""

    if not 0 <= inward_count <= num_spacecraft - 1:
        raise ValueError("inward_count must lie in [0, num_spacecraft-1].")
    inward = [k * separation_rad for k in range(1, inward_count + 1)]
    outward_count = num_spacecraft - 1 - inward_count
    outward = [-k * separation_rad for k in range(1, outward_count + 1)]
    return np.asarray([0.0, *inward, *outward], dtype=float)


def full_fov_clearance_rad(
    boresights: np.ndarray,
    iv_axes: np.ndarray,
    protected_half_angle_rad: float,
) -> np.ndarray:
    """Positive when the complete FOV clears the expanded IV-zone boundary."""

    centre_separation = angular_separation_rad(boresights, iv_axes)
    return centre_separation - float(protected_half_angle_rad)



def minimum_outward_shift_rad(
    nominal: np.ndarray,
    iv_axis: np.ndarray,
    protected_half_angle_rad: float,
    buffer_rad: float,
) -> float:
    """Return the minimum non-positive offset that clears the IV zone.

    The positive tangent direction points from the nominal boresight toward the
    IV axis.  Therefore a negative signed offset moves away from the IV zone.
    The returned value is zero when the nominal complete FOV is already clear.
    """

    alpha = float(angular_separation_rad(
        _unit(nominal, "nominal").reshape(1, 3),
        _unit(iv_axis, "IV axis").reshape(1, 3),
    )[0])
    required = float(protected_half_angle_rad)
    if alpha >= required:
        return 0.0
    return alpha - required - float(buffer_rad)


def _pairwise_clear(
    candidate: np.ndarray,
    assigned: list[np.ndarray],
    required_separation_rad: float,
    tolerance_rad: float = 1.0e-12,
) -> bool:
    if not assigned:
        return True
    existing = np.asarray(assigned, dtype=float)
    candidate_rows = np.broadcast_to(
        _unit(candidate, "candidate boresight").reshape(1, 3),
        existing.shape,
    )
    separations = angular_separation_rad(candidate_rows, existing)
    return bool(np.all(separations >= float(required_separation_rad) - tolerance_rad))



def minimum_pairwise_separation_rad(boresights: np.ndarray) -> float:
    values = _normalize_rows(np.asarray(boresights, dtype=float), "boresights")
    count = values.shape[0]
    if count < 2:
        return float("nan")
    minimum = np.pi
    for first in range(count - 1):
        a = np.broadcast_to(values[first].reshape(1, 3), (count - first - 1, 3))
        separations = angular_separation_rad(a, values[first + 1 :])
        minimum = min(minimum, float(np.min(separations)))
    return minimum


def _offset_clears_iv(
    nominal: np.ndarray,
    tangent: np.ndarray,
    offset_rad: float,
    iv_axis: np.ndarray,
    protected_half_angle_rad: float,
) -> tuple[bool, np.ndarray, float]:
    boresight = boresight_on_tangent(nominal, tangent, float(offset_rad))
    clearance = float(full_fov_clearance_rad(
        boresight.reshape(1, 3),
        _unit(iv_axis, "IV axis").reshape(1, 3),
        protected_half_angle_rad,
    )[0])
    return clearance >= -1.0e-12, boresight, clearance


# =============================================================================
# LPF/config loading and formation construction
# =============================================================================


def _resolve_relative_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_configuration(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"YAML root must be a mapping: {config_path}")
    return config


def requested_phases_deg(num_spacecraft: int) -> np.ndarray:
    if RELATIVE_PHASE_DEG is None:
        return np.linspace(0.0, 360.0, num_spacecraft, endpoint=False)

    phases = np.asarray(RELATIVE_PHASE_DEG, dtype=float)
    if phases.shape != (num_spacecraft,):
        raise ValueError(
            "RELATIVE_PHASE_DEG must contain exactly one value per spacecraft: "
            f"expected {num_spacecraft}, got {phases.shape}."
        )
    if np.any(~np.isfinite(phases)):
        raise ValueError("RELATIVE_PHASE_DEG must contain finite values.")
    phases = np.mod(phases, 360.0)
    if not np.isclose(phases[0], 0.0, atol=1.0e-12):
        raise ValueError("The first relative phase must be 0 degrees for SC1.")
    if np.unique(np.round(phases, decimals=12)).size != num_spacecraft:
        raise ValueError("Relative spacecraft phases must be distinct.")
    return phases


def load_lpf_study_geometry(config_path: Path, config: dict) -> StudyGeometry:
    orbit_path = _resolve_relative_path(config["orbit_file_path"], config_path)
    if not orbit_path.is_file():
        raise FileNotFoundError(
            f"LPF orbit file not found: {orbit_path}\n"
            "Place the configured LPF CSV relative to the YAML or update orbit_file_path."
        )

    orbit = pd.read_csv(
        orbit_path,
        sep=",",
        header=0,
        names=config.get("orbit_column_names"),
    )

    period_start = int(config["quasi_halo_start"])
    period_end = int(config["quasi_halo_one_period_end"])
    if not 0 <= period_start < period_end <= len(orbit):
        raise ValueError(
            "Invalid one-period LPF slice: "
            f"start={period_start}, end={period_end}, rows={len(orbit)}."
        )

    period = orbit.iloc[period_start:period_end].reset_index(drop=True)
    period_length = len(period)

    required = [
        "SUN_EARTH_CO_X_(km)",
        "SUN_EARTH_CO_Y_(km)",
        "SUN_EARTH_CO_Z_(km)",
        "MOON_SUN_EARTH_CO_X_(km)",
        "MOON_SUN_EARTH_CO_Y_(km)",
        "MOON_SUN_EARTH_CO_Z_(km)",
    ]
    missing = [column for column in required if column not in period.columns]
    if missing:
        raise KeyError(f"LPF orbit is missing required SECR columns: {missing}")

    num_spacecraft = (
        int(NUM_SPACECRAFT_OVERRIDE)
        if NUM_SPACECRAFT_OVERRIDE is not None
        else int(config["num_spacecraft"])
    )
    if num_spacecraft < 1:
        raise ValueError("The study requires at least one spacecraft.")

    phases_deg = requested_phases_deg(num_spacecraft)
    phase_offsets = np.rint(phases_deg / 360.0 * period_length).astype(int)
    phase_offsets %= period_length
    if np.unique(phase_offsets).size != num_spacecraft:
        raise ValueError(
            "Two requested phases map to the same LPF sample. Use more widely "
            "separated phases or a more finely sampled orbit."
        )
    realized_phases = phase_offsets / period_length * 360.0

    stride = int(DESIGN_SAMPLE_STRIDE)
    if stride <= 0:
        raise ValueError("DESIGN_SAMPLE_STRIDE must be positive.")
    source_indices = np.arange(0, period_length, stride, dtype=int)
    if source_indices[-1] != period_length - 1:
        source_indices = np.append(source_indices, period_length - 1)

    anchor = int(ANCHOR_PERIOD_INDEX) % period_length
    position_columns = [
        "SUN_EARTH_CO_X_(km)",
        "SUN_EARTH_CO_Y_(km)",
        "SUN_EARTH_CO_Z_(km)",
    ]
    all_positions = period[position_columns].to_numpy(dtype=float)

    # Preserve the complete one-period tracks for the 3-D orbit scenes.  The
    # sampled design positions below are drawn directly from these tracks, so
    # the plotted spacecraft locations and the boresight solution are exactly
    # consistent.
    full_period_indices = np.arange(period_length, dtype=int)
    sc_orbit_tracks = np.empty((period_length, num_spacecraft, 3), dtype=float)
    for sc_index, phase_offset in enumerate(phase_offsets):
        row_indices = (
            full_period_indices + anchor + int(phase_offset)
        ) % period_length
        sc_orbit_tracks[:, sc_index, :] = all_positions[row_indices]

    sc_positions = sc_orbit_tracks[source_indices]

    moon_columns = [
        "MOON_SUN_EARTH_CO_X_(km)",
        "MOON_SUN_EARTH_CO_Y_(km)",
        "MOON_SUN_EARTH_CO_Z_(km)",
    ]
    moon_orbit_track = period[moon_columns].to_numpy(dtype=float)
    moon_positions = moon_orbit_track[source_indices]

    if "Time" in period.columns:
        times = period.iloc[source_indices]["Time"].astype(str).to_numpy()
    else:
        times = source_indices.astype(str)

    earth_positions = np.zeros((source_indices.size, 3), dtype=float)
    iv_axes = np.empty_like(sc_positions)
    iv_half_angle_rad = float(np.deg2rad(IV_HALF_ANGLE_DEG))

    for sc_index in range(num_spacecraft):
        zone = compute_earth_moon_invisibility_zone_batch(
            spacecraft_position=sc_positions[:, sc_index, :],
            earth_position=earth_positions,
            moon_position=moon_positions,
            half_angle_deg=IV_HALF_ANGLE_DEG,
        )
        iv_axes[:, sc_index, :] = np.asarray(zone.axis_geo_eme, dtype=float)
        iv_half_angle_rad = float(zone.half_angle_rad)

    return StudyGeometry(
        times=times,
        source_indices=source_indices,
        spacecraft_positions_secr_km=sc_positions,
        moon_positions_secr_km=moon_positions,
        spacecraft_orbit_tracks_secr_km=sc_orbit_tracks,
        moon_orbit_track_secr_km=moon_orbit_track,
        iv_axes_secr=iv_axes,
        iv_half_angle_rad=iv_half_angle_rad,
        requested_phase_deg=phases_deg,
        realized_phase_deg=realized_phases,
        phase_offset_indices=phase_offsets,
    )


# =============================================================================
# Packing modes
# =============================================================================


def _mean_unit_axis(axes: np.ndarray) -> np.ndarray:
    return _unit(np.sum(_normalize_rows(axes, "IV axes"), axis=0), "mean IV axis")


def fixed_tangent(geometry: StudyGeometry, nominal: np.ndarray) -> np.ndarray:
    source = str(FIXED_TANGENT_SOURCE).strip().lower()
    if source == "initial":
        axis = _mean_unit_axis(geometry.iv_axes_secr[0])
    elif source == "mean":
        axis = _mean_unit_axis(geometry.iv_axes_secr.reshape(-1, 3))
    else:
        raise ValueError("FIXED_TANGENT_SOURCE must be 'initial' or 'mean'.")
    return tangent_toward_axis(nominal, axis)


def solve_fixed_global(
    geometry: StudyGeometry,
    nominal: np.ndarray,
    separation_rad: float,
    protected_half_angle_rad: float,
) -> PackingResult:
    """Pack one fixed SECR pattern using one common fixed tangent.

    The tangent comes from ``FIXED_TANGENT_SOURCE``.  Every spacecraft offset
    is static over the study period, but each candidate is checked against that
    spacecraft's own IV zone at every sampled epoch.  If SC1's nominal FOV is
    not globally feasible, SC1 is moved outward along the fixed tangent until
    all epochs clear.  The remaining spacecraft are then packed inward first;
    after the first inward failure, all remaining spacecraft are packed
    outward from the nominal direction.
    """

    epochs, num_spacecraft, _ = geometry.iv_axes_secr.shape
    tangent = fixed_tangent(geometry, nominal)
    tangent_time_sc = np.broadcast_to(
        tangent.reshape(1, 1, 3),
        (epochs, num_spacecraft, 3),
    ).copy()

    b0 = _unit(nominal, "nominal")
    nominal_time = np.broadcast_to(
        b0.reshape(1, 1, 3),
        (epochs, num_spacecraft, 3),
    )
    nominal_clearance = full_fov_clearance_rad(
        nominal_time,
        geometry.iv_axes_secr,
        protected_half_angle_rad,
    )

    policy = str(NOMINAL_VIOLATION_POLICY).strip().lower()
    if policy not in {"shift_outward", "fail"}:
        raise ValueError(
            "NOMINAL_VIOLATION_POLICY must be 'shift_outward' or 'fail'."
        )

    step_rad = float(np.deg2rad(PACKING_SEARCH_STEP_DEG))
    max_offset_rad = float(np.deg2rad(MAX_OUTWARD_OFFSET_DEG))
    if step_rad <= 0.0 or max_offset_rad <= 0.0:
        raise ValueError(
            "PACKING_SEARCH_STEP_DEG and MAX_OUTWARD_OFFSET_DEG must be positive."
        )

    def evaluate(sc_index: int, offset_rad: float) -> tuple[bool, np.ndarray, np.ndarray]:
        candidate = boresight_on_tangent(b0, tangent, float(offset_rad))
        candidate_time = np.broadcast_to(candidate.reshape(1, 3), (epochs, 3))
        candidate_clearance = full_fov_clearance_rad(
            candidate_time,
            geometry.iv_axes_secr[:, sc_index, :],
            protected_half_angle_rad,
        )
        return (
            bool(np.all(candidate_clearance >= -1.0e-12)),
            candidate,
            candidate_clearance,
        )

    assigned_vectors: list[np.ndarray] = []
    selected_offsets = np.empty(num_spacecraft, dtype=float)
    selected_static = np.empty((num_spacecraft, 3), dtype=float)
    selected_clearance = np.empty((epochs, num_spacecraft), dtype=float)
    minimum_shifts_static = np.empty(num_spacecraft, dtype=float)

    # Find the minimum fixed outward shift needed by each spacecraft on the
    # chosen fixed tangent.  This is also used as the outward-search floor.
    for sc_index in range(num_spacecraft):
        minimum_shift = np.nan
        for candidate_offset in -np.arange(
            0.0,
            max_offset_rad + 0.5 * step_rad,
            step_rad,
        ):
            ok, _, _ = evaluate(sc_index, float(candidate_offset))
            if ok:
                minimum_shift = float(candidate_offset)
                break
        if not np.isfinite(minimum_shift):
            raise RuntimeError(
                "No fixed outward correction clears all sampled epochs for "
                f"SC{sc_index + 1} within {MAX_OUTWARD_OFFSET_DEG:.3f} deg."
            )
        minimum_shifts_static[sc_index] = minimum_shift

    if policy == "fail" and np.any(nominal_clearance < -1.0e-12):
        epoch_index, sc_index = np.unravel_index(
            int(np.argmin(nominal_clearance)),
            nominal_clearance.shape,
        )
        raise RuntimeError(
            "Nominal complete-FOV violation in fixed mode at sampled epoch "
            f"{epoch_index}, SC{sc_index + 1}: "
            f"{np.rad2deg(nominal_clearance[epoch_index, sc_index]):.6f} deg."
        )

    # SC1 anchors the fixed ordering.
    sc1_offset = 0.0 if np.all(nominal_clearance[:, 0] >= -1.0e-12) else float(minimum_shifts_static[0])
    ok, sc1_boresight, sc1_clearance = evaluate(0, sc1_offset)
    if not ok:
        raise RuntimeError("SC1 fixed outward correction did not clear all epochs.")
    selected_offsets[0] = sc1_offset
    selected_static[0] = sc1_boresight
    selected_clearance[:, 0] = sc1_clearance
    assigned_vectors.append(sc1_boresight)

    inward_open = True
    inward_slot = 1
    outward_slot = 1
    inward_count = 0

    for sc_index in range(1, num_spacecraft):
        placed = False

        if inward_open:
            start_inward = inward_slot * float(separation_rad)
            candidate_offset = start_inward
            while candidate_offset <= max_offset_rad + 1.0e-12:
                ok, candidate_boresight, candidate_clearance = evaluate(
                    sc_index,
                    candidate_offset,
                )
                if ok and _pairwise_clear(
                    candidate_boresight,
                    assigned_vectors,
                    separation_rad,
                ):
                    selected_offsets[sc_index] = candidate_offset
                    selected_static[sc_index] = candidate_boresight
                    selected_clearance[:, sc_index] = candidate_clearance
                    assigned_vectors.append(candidate_boresight)
                    inward_count += 1
                    inward_slot += 1
                    placed = True
                    break
                candidate_offset += step_rad

            if not placed:
                inward_open = False

        if not placed:
            desired_outward = -outward_slot * float(separation_rad)
            start_outward = min(desired_outward, float(minimum_shifts_static[sc_index]))
            candidate_offset = start_outward
            while candidate_offset >= -max_offset_rad - 1.0e-12:
                ok, candidate_boresight, candidate_clearance = evaluate(
                    sc_index,
                    candidate_offset,
                )
                if ok and _pairwise_clear(
                    candidate_boresight,
                    assigned_vectors,
                    separation_rad,
                ):
                    selected_offsets[sc_index] = candidate_offset
                    selected_static[sc_index] = candidate_boresight
                    selected_clearance[:, sc_index] = candidate_clearance
                    assigned_vectors.append(candidate_boresight)
                    outward_slot += 1
                    placed = True
                    break
                candidate_offset -= step_rad

        if not placed:
            raise RuntimeError(
                "No fixed outward placement found for "
                f"SC{sc_index + 1} within {MAX_OUTWARD_OFFSET_DEG:.3f} deg."
            )

    selected_time = np.broadcast_to(
        selected_static.reshape(1, num_spacecraft, 3),
        (epochs, num_spacecraft, 3),
    ).copy()
    minimum_shifts = np.broadcast_to(
        minimum_shifts_static.reshape(1, num_spacecraft),
        (epochs, num_spacecraft),
    ).copy()

    return PackingResult(
        mode="fixed_global",
        boresights_secr=selected_time,
        offsets_rad=np.broadcast_to(
            selected_offsets.reshape(1, num_spacecraft),
            (epochs, num_spacecraft),
        ).copy(),
        inward_count=np.full(epochs, inward_count, dtype=int),
        iv_clearance_rad=selected_clearance,
        tangent_secr=tangent_time_sc,
        nominal_clearance_rad=nominal_clearance,
        minimum_outward_shift_rad=minimum_shifts,
    )


def solve_dynamic_per_spacecraft(
    geometry: StudyGeometry,
    nominal: np.ndarray,
    separation_rad: float,
    protected_half_angle_rad: float,
) -> PackingResult:
    """Pack boresights at every epoch using each spacecraft's own IV geometry.

    Positive offsets point from the nominal boresight toward that spacecraft's
    instantaneous IV axis.  Negative offsets point away from it.

    SC1 is assigned first.  It remains at zero whenever feasible; otherwise it
    receives the minimum analytical outward correction.  Subsequent spacecraft
    are attempted on successive inward slots.  After the first inward failure,
    all remaining spacecraft are placed outward from the nominal direction.
    Every accepted placement satisfies both that spacecraft's complete-FOV IV
    clearance and the actual angular separation from all already assigned FOVs.
    """

    epochs, num_spacecraft, _ = geometry.iv_axes_secr.shape
    boresights = np.empty((epochs, num_spacecraft, 3), dtype=float)
    offsets = np.empty((epochs, num_spacecraft), dtype=float)
    inward_counts = np.empty(epochs, dtype=int)
    clearance = np.empty((epochs, num_spacecraft), dtype=float)
    tangents = np.empty((epochs, num_spacecraft, 3), dtype=float)
    nominal_clearance = np.empty((epochs, num_spacecraft), dtype=float)
    minimum_shifts = np.empty((epochs, num_spacecraft), dtype=float)

    policy = str(NOMINAL_VIOLATION_POLICY).strip().lower()
    if policy not in {"shift_outward", "fail"}:
        raise ValueError(
            "NOMINAL_VIOLATION_POLICY must be 'shift_outward' or 'fail'."
        )

    search_step_rad = float(np.deg2rad(PACKING_SEARCH_STEP_DEG))
    max_outward_rad = float(np.deg2rad(MAX_OUTWARD_OFFSET_DEG))
    boundary_buffer_rad = float(np.deg2rad(OUTWARD_CLEARANCE_BUFFER_DEG))
    if search_step_rad <= 0.0:
        raise ValueError("PACKING_SEARCH_STEP_DEG must be positive.")
    if max_outward_rad <= 0.0:
        raise ValueError("MAX_OUTWARD_OFFSET_DEG must be positive.")
    if boundary_buffer_rad < 0.0:
        raise ValueError("OUTWARD_CLEARANCE_BUFFER_DEG cannot be negative.")

    b0 = _unit(nominal, "nominal")

    for epoch_index in range(epochs):
        epoch_axes = geometry.iv_axes_secr[epoch_index]
        epoch_tangents = np.empty((num_spacecraft, 3), dtype=float)
        epoch_min_shifts = np.empty(num_spacecraft, dtype=float)
        epoch_nominal_clearance = np.empty(num_spacecraft, dtype=float)

        for sc_index in range(num_spacecraft):
            axis = epoch_axes[sc_index]
            tangent = tangent_toward_axis(b0, axis)
            epoch_tangents[sc_index] = tangent
            epoch_nominal_clearance[sc_index] = float(
                full_fov_clearance_rad(
                    b0.reshape(1, 3),
                    axis.reshape(1, 3),
                    protected_half_angle_rad,
                )[0]
            )
            epoch_min_shifts[sc_index] = minimum_outward_shift_rad(
                b0,
                axis,
                protected_half_angle_rad,
                boundary_buffer_rad,
            )

        tangents[epoch_index] = epoch_tangents
        nominal_clearance[epoch_index] = epoch_nominal_clearance
        minimum_shifts[epoch_index] = epoch_min_shifts

        if policy == "fail" and np.any(epoch_nominal_clearance < -1.0e-12):
            bad_sc = int(np.argmin(epoch_nominal_clearance))
            raise RuntimeError(
                "Nominal complete-FOV violation at sampled epoch "
                f"{epoch_index}, SC{bad_sc + 1}: "
                f"{np.rad2deg(epoch_nominal_clearance[bad_sc]):.6f} deg."
            )

        assigned_vectors: list[np.ndarray] = []
        epoch_offsets = np.empty(num_spacecraft, dtype=float)
        epoch_boresights = np.empty((num_spacecraft, 3), dtype=float)
        epoch_clearance = np.empty(num_spacecraft, dtype=float)

        # SC1 anchors the ordering.  Keep nominal if possible; otherwise move
        # only as far outward as required by SC1's own instantaneous IV zone.
        sc1_offset = (
            0.0
            if epoch_nominal_clearance[0] >= -1.0e-12
            else float(epoch_min_shifts[0])
        )
        ok, sc1_boresight, sc1_clearance = _offset_clears_iv(
            b0,
            epoch_tangents[0],
            sc1_offset,
            epoch_axes[0],
            protected_half_angle_rad,
        )
        if not ok:
            raise RuntimeError(
                f"SC1 outward correction failed at sampled epoch {epoch_index}."
            )
        epoch_offsets[0] = sc1_offset
        epoch_boresights[0] = sc1_boresight
        epoch_clearance[0] = sc1_clearance
        assigned_vectors.append(sc1_boresight)

        inward_open = True
        inward_slot = 1
        outward_slot = 1
        inward_count = 0

        for sc_index in range(1, num_spacecraft):
            axis = epoch_axes[sc_index]
            tangent = epoch_tangents[sc_index]
            placed = False

            if inward_open:
                # Search from the next compact inward slot toward the IV
                # boundary.  The upper bound is obtained analytically from this
                # spacecraft's own IV geometry.
                alpha = float(angular_separation_rad(
                    b0.reshape(1, 3),
                    axis.reshape(1, 3),
                )[0])
                max_inward_offset = alpha - float(protected_half_angle_rad)
                start_offset = inward_slot * float(separation_rad)

                if start_offset <= max_inward_offset + 1.0e-12:
                    candidate_offsets = np.arange(
                        start_offset,
                        max_inward_offset + 0.5 * search_step_rad,
                        search_step_rad,
                    )
                    if candidate_offsets.size == 0:
                        candidate_offsets = np.asarray([start_offset])
                    for candidate_offset in candidate_offsets:
                        ok, candidate_boresight, candidate_clearance = _offset_clears_iv(
                            b0,
                            tangent,
                            float(candidate_offset),
                            axis,
                            protected_half_angle_rad,
                        )
                        if ok and _pairwise_clear(
                            candidate_boresight,
                            assigned_vectors,
                            separation_rad,
                        ):
                            epoch_offsets[sc_index] = float(candidate_offset)
                            epoch_boresights[sc_index] = candidate_boresight
                            epoch_clearance[sc_index] = candidate_clearance
                            assigned_vectors.append(candidate_boresight)
                            inward_count += 1
                            inward_slot += 1
                            placed = True
                            break

                if not placed:
                    # This implements the requested rule: once the next inward
                    # spacecraft cannot fit, every remaining spacecraft goes to
                    # the outward side of the nominal direction.
                    inward_open = False

            if not placed:
                desired_outward = -outward_slot * float(separation_rad)
                # If the nominal FOV itself is infeasible for this spacecraft,
                # begin no closer to zero than its minimum outward correction.
                start_outward = min(desired_outward, float(epoch_min_shifts[sc_index]))

                candidate_offset = start_outward
                while candidate_offset >= -max_outward_rad - 1.0e-12:
                    ok, candidate_boresight, candidate_clearance = _offset_clears_iv(
                        b0,
                        tangent,
                        candidate_offset,
                        axis,
                        protected_half_angle_rad,
                    )
                    if ok and _pairwise_clear(
                        candidate_boresight,
                        assigned_vectors,
                        separation_rad,
                    ):
                        epoch_offsets[sc_index] = float(candidate_offset)
                        epoch_boresights[sc_index] = candidate_boresight
                        epoch_clearance[sc_index] = candidate_clearance
                        assigned_vectors.append(candidate_boresight)
                        outward_slot += 1
                        placed = True
                        break
                    candidate_offset -= search_step_rad

            if not placed:
                raise RuntimeError(
                    "No outward placement found at sampled epoch "
                    f"{epoch_index} for SC{sc_index + 1} within "
                    f"{MAX_OUTWARD_OFFSET_DEG:.3f} deg of nominal."
                )

        inward_counts[epoch_index] = inward_count
        offsets[epoch_index] = epoch_offsets
        boresights[epoch_index] = epoch_boresights
        clearance[epoch_index] = epoch_clearance

    return PackingResult(
        mode="dynamic_per_spacecraft",
        boresights_secr=boresights,
        offsets_rad=offsets,
        inward_count=inward_counts,
        iv_clearance_rad=clearance,
        tangent_secr=tangents,
        nominal_clearance_rad=nominal_clearance,
        minimum_outward_shift_rad=minimum_shifts,
    )


# =============================================================================
# Cone cross sections and visualization
# =============================================================================


def perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _unit(axis, "cone axis")
    trial = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(a, trial))) > 0.90:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    u = _unit(np.cross(a, trial), "cone basis u")
    v = _unit(np.cross(a, u), "cone basis v")
    return u, v


def cone_plane_footprint_yz(
    apex: np.ndarray,
    axis: np.ndarray,
    half_angle_rad: float,
    x_plane_km: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect cone boundary rays with x=constant and return y/z coordinates."""

    apex = np.asarray(apex, dtype=float).reshape(3)
    axis = _unit(axis, "cone axis")
    u, v = perpendicular_basis(axis)
    phi = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=True)
    directions = (
        np.cos(half_angle_rad) * axis[None, :]
        + np.sin(half_angle_rad)
        * (np.cos(phi)[:, None] * u[None, :] + np.sin(phi)[:, None] * v[None, :])
    )

    dx = directions[:, 0]
    valid = np.abs(dx) > 1.0e-14
    scale = np.full(phi.shape, np.nan, dtype=float)
    scale[valid] = (float(x_plane_km) - apex[0]) / dx[valid]
    valid &= scale > 0.0

    points = np.full((phi.size, 3), np.nan, dtype=float)
    points[valid] = apex[None, :] + scale[valid, None] * directions[valid]
    return points[:, 1], points[:, 2]


def boresight_centre_on_plane(
    apex: np.ndarray,
    boresight: np.ndarray,
    x_plane_km: float,
) -> tuple[float, float] | None:
    apex = np.asarray(apex, dtype=float).reshape(3)
    direction = _unit(boresight, "boresight")
    if abs(float(direction[0])) <= 1.0e-14:
        return None
    scale = (float(x_plane_km) - apex[0]) / direction[0]
    if scale <= 0.0:
        return None
    point = apex + scale * direction
    return float(point[1]), float(point[2])


def snapshot_indices(count: int) -> np.ndarray:
    values = []
    for fraction in SNAPSHOT_FRACTIONS:
        f = float(fraction)
        if not 0.0 <= f <= 1.0:
            raise ValueError("SNAPSHOT_FRACTIONS must lie in [0,1].")
        values.append(int(round(f * (count - 1))))
    return np.unique(np.asarray(values, dtype=int))



def three_d_snapshot_indices(geometry: StudyGeometry) -> np.ndarray:
    """Resolve the user-selected 3-D scene epochs to sampled epoch indices."""

    mode = str(THREE_D_EPOCH_SELECTION_MODE).strip().lower()
    values = tuple(THREE_D_EPOCH_VALUES)
    if not values:
        return np.empty(0, dtype=int)

    resolved: list[int] = []
    sample_count = len(geometry.times)
    period_count = geometry.spacecraft_orbit_tracks_secr_km.shape[0]

    if mode == "sampled_index":
        for value in values:
            index = int(value)
            if not 0 <= index < sample_count:
                raise ValueError(
                    "THREE_D_EPOCH_VALUES contains sampled index "
                    f"{index}, but the valid range is 0 to {sample_count - 1}."
                )
            resolved.append(index)

    elif mode == "source_period_index":
        for value in values:
            source_index = int(value)
            if not 0 <= source_index < period_count:
                raise ValueError(
                    "THREE_D_EPOCH_VALUES contains source-period index "
                    f"{source_index}, but the valid range is 0 to "
                    f"{period_count - 1}."
                )
            nearest = int(
                np.argmin(np.abs(geometry.source_indices - source_index))
            )
            resolved.append(nearest)
            actual = int(geometry.source_indices[nearest])
            if actual != source_index:
                print(
                    "3-D epoch request source-period index "
                    f"{source_index} maps to nearest sampled index {nearest} "
                    f"(source-period index {actual})."
                )

    elif mode == "fraction":
        for value in values:
            fraction = float(value)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    "THREE_D_EPOCH_VALUES must lie in [0,1] when "
                    "THREE_D_EPOCH_SELECTION_MODE='fraction'."
                )
            resolved.append(int(round(fraction * (sample_count - 1))))

    else:
        raise ValueError(
            "THREE_D_EPOCH_SELECTION_MODE must be 'sampled_index', "
            "'source_period_index', or 'fraction'."
        )

    # Preserve the requested order while removing duplicates.
    return np.asarray(list(dict.fromkeys(resolved)), dtype=int)


def cone_surface_xyz(
    apex: np.ndarray,
    axis: np.ndarray,
    half_angle_rad: float,
    length_km: float,
    azimuth_samples: int,
    axial_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a finite cone surface for 3-D visualization only."""

    apex = np.asarray(apex, dtype=float).reshape(3)
    axis = _unit(axis, "cone axis")
    length = float(length_km)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("THREE_D_CONE_LENGTH_KM must be finite and positive.")
    if int(azimuth_samples) < 8 or int(axial_samples) < 2:
        raise ValueError(
            "3-D cone sampling requires at least 8 azimuth and 2 axial samples."
        )

    u, v = perpendicular_basis(axis)
    phi = np.linspace(0.0, 2.0 * np.pi, int(azimuth_samples), endpoint=True)
    distance = np.linspace(0.0, length, int(axial_samples), endpoint=True)
    radial_direction = (
        np.cos(phi)[:, None] * u[None, :]
        + np.sin(phi)[:, None] * v[None, :]
    )
    centres = apex[None, :] + distance[:, None] * axis[None, :]
    radii = distance * np.tan(float(half_angle_rad))
    points = (
        centres[:, None, :]
        + radii[:, None, None] * radial_direction[None, :, :]
    )
    return points[:, :, 0], points[:, :, 1], points[:, :, 2]


def _set_equal_3d_limits(ax: object, points: np.ndarray, padding_fraction: float = 0.05) -> None:
    """Apply equal physical scaling to x, y, and z on a Matplotlib 3-D axis."""

    values = np.asarray(points, dtype=float).reshape(-1, 3)
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.size == 0:
        return
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    centre = 0.5 * (minimum + maximum)
    half_range = 0.5 * float(np.max(maximum - minimum))
    if half_range <= 0.0:
        half_range = 1.0
    half_range *= 1.0 + 2.0 * float(padding_fraction)
    ax.set_xlim(centre[0] - half_range, centre[0] + half_range)
    ax.set_ylim(centre[1] - half_range, centre[1] + half_range)
    ax.set_zlim(centre[2] - half_range, centre[2] + half_range)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def save_3d_orbit_cone_figures(
    output_dir: Path,
    geometry: StudyGeometry,
    result: PackingResult,
    fov_half_angle_rad: float,
    show: bool,
) -> list[Path]:
    """Plot the LPF orbit, bodies, spacecraft, and epoch-specific cone geometry."""

    saved: list[Path] = []
    epoch_indices = three_d_snapshot_indices(geometry)
    if epoch_indices.size == 0:
        return saved

    decimation = int(THREE_D_ORBIT_DECIMATION)
    if decimation <= 0:
        raise ValueError("THREE_D_ORBIT_DECIMATION must be positive.")

    reference_orbit = geometry.spacecraft_orbit_tracks_secr_km[::decimation, 0, :]
    moon_orbit = geometry.moon_orbit_track_secr_km[::decimation]
    num_spacecraft = geometry.spacecraft_positions_secr_km.shape[1]
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])

    for epoch_index in epoch_indices:
        positions = geometry.spacecraft_positions_secr_km[epoch_index]
        current_moon = geometry.moon_positions_secr_km[epoch_index]

        fig = plt.figure(figsize=(11.0, 8.5))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            reference_orbit[:, 0],
            reference_orbit[:, 1],
            reference_orbit[:, 2],
            linewidth=1.3,
            label="LPF quasi-halo orbit",
        )
        ax.plot(
            moon_orbit[:, 0],
            moon_orbit[:, 1],
            moon_orbit[:, 2],
            linewidth=1.0,
            linestyle="--",
            label="Moon orbit",
        )
        ax.scatter(0.0, 0.0, 0.0, marker="o", s=60, label="Earth")
        ax.scatter(
            current_moon[0],
            current_moon[1],
            current_moon[2],
            marker="o",
            s=42,
            label="Moon at epoch",
        )

        scene_points = [reference_orbit, moon_orbit, np.zeros((1, 3)), positions]

        for sc_index in range(num_spacecraft):
            colour = palette[sc_index % len(palette)]
            apex = positions[sc_index]
            boresight = result.boresights_secr[epoch_index, sc_index]
            iv_axis = geometry.iv_axes_secr[epoch_index, sc_index]

            ax.scatter(
                apex[0], apex[1], apex[2],
                marker="^", s=55, color=colour,
            )
            ax.text(
                apex[0], apex[1], apex[2],
                f"  SC{sc_index + 1}",
                color=colour,
            )

            x_fov, y_fov, z_fov = cone_surface_xyz(
                apex,
                boresight,
                fov_half_angle_rad,
                THREE_D_CONE_LENGTH_KM,
                THREE_D_CONE_AZIMUTH_SAMPLES,
                THREE_D_CONE_AXIAL_SAMPLES,
            )
            ax.plot_surface(
                x_fov,
                y_fov,
                z_fov,
                color=colour,
                alpha=0.22,
                linewidth=0.0,
                antialiased=True,
                shade=False,
            )
            fov_end = apex + float(THREE_D_CONE_LENGTH_KM) * boresight
            ax.plot(
                [apex[0], fov_end[0]],
                [apex[1], fov_end[1]],
                [apex[2], fov_end[2]],
                color=colour,
                linewidth=1.6,
            )
            scene_points.append(np.column_stack((x_fov[-1], y_fov[-1], z_fov[-1])))

            if THREE_D_SHOW_IV_CONES:
                x_iv, y_iv, z_iv = cone_surface_xyz(
                    apex,
                    iv_axis,
                    geometry.iv_half_angle_rad,
                    THREE_D_CONE_LENGTH_KM,
                    THREE_D_CONE_AZIMUTH_SAMPLES,
                    max(2, int(THREE_D_CONE_AXIAL_SAMPLES // 2)),
                )
                ax.plot_wireframe(
                    x_iv,
                    y_iv,
                    z_iv,
                    color=colour,
                    alpha=0.28,
                    linewidth=0.55,
                    rstride=1,
                    cstride=max(1, int(THREE_D_CONE_AZIMUTH_SAMPLES // 12)),
                )
                iv_end = apex + float(THREE_D_CONE_LENGTH_KM) * iv_axis
                ax.plot(
                    [apex[0], iv_end[0]],
                    [apex[1], iv_end[1]],
                    [apex[2], iv_end[2]],
                    color=colour,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                )
                scene_points.append(np.column_stack((x_iv[-1], y_iv[-1], z_iv[-1])))

            if THREE_D_SHOW_EARTH_MOON_LOS:
                ax.plot(
                    [apex[0], 0.0],
                    [apex[1], 0.0],
                    [apex[2], 0.0],
                    color=colour,
                    linewidth=0.6,
                    alpha=0.35,
                )
                ax.plot(
                    [apex[0], current_moon[0]],
                    [apex[1], current_moon[1]],
                    [apex[2], current_moon[2]],
                    color=colour,
                    linewidth=0.6,
                    linestyle=":",
                    alpha=0.35,
                )

        source_index = int(geometry.source_indices[epoch_index])
        ax.set_xlabel("SECR x (km)")
        ax.set_ylabel("SECR y (km)")
        ax.set_zlabel("SECR z (km)")
        ax.set_title(
            f"{result.mode}: sampled epoch {epoch_index}, "
            f"source-period index {source_index}\n"
            f"time={geometry.times[epoch_index]}"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
        _set_equal_3d_limits(ax, np.vstack(scene_points))
        fig.tight_layout()

        path = output_dir / (
            f"orbit_fov_cones_3d_{result.mode}_"
            f"epoch-{epoch_index:04d}_source-{source_index:06d}.svg"
        )
        fig.savefig(path, format="svg", bbox_inches="tight")
        saved.append(path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved


def save_projection_figures(
    output_dir: Path,
    geometry: StudyGeometry,
    result: PackingResult,
    fov_half_angle_rad: float,
    show: bool,
) -> list[Path]:
    saved: list[Path] = []
    num_spacecraft = geometry.spacecraft_positions_secr_km.shape[1]

    for epoch_index in snapshot_indices(len(geometry.times)):
        positions = geometry.spacecraft_positions_secr_km[epoch_index]
        mean_x = float(np.mean(positions[:, 0]))

        nominal_x = float(_unit(NOMINAL_BORESIGHT_SECR, "nominal")[0])
        if abs(nominal_x) <= 1.0e-12:
            raise ValueError(
                "YZ cross sections require an x-aligned nominal boresight."
            )

        for distance_km in CROSS_SECTION_DISTANCES_KM:
            distance_km = float(distance_km)
            if not np.isfinite(distance_km) or distance_km <= 0.0:
                raise ValueError(
                    "CROSS_SECTION_DISTANCES_KM values must be finite and positive."
                )
            x_plane = mean_x + distance_km * nominal_x

            fig, ax = plt.subplots(figsize=(8.0, 7.0))
            for sc_index in range(num_spacecraft):
                apex = positions[sc_index]
                boresight = result.boresights_secr[epoch_index, sc_index]
                iv_axis = geometry.iv_axes_secr[epoch_index, sc_index]

                y_fov, z_fov = cone_plane_footprint_yz(
                    apex,
                    boresight,
                    fov_half_angle_rad,
                    x_plane,
                    CONE_BOUNDARY_SAMPLES,
                )
                line = ax.plot(
                    y_fov,
                    z_fov,
                    linewidth=1.6,
                    label=f"SC{sc_index + 1} FOV",
                )[0]

                y_iv, z_iv = cone_plane_footprint_yz(
                    apex,
                    iv_axis,
                    geometry.iv_half_angle_rad,
                    x_plane,
                    CONE_BOUNDARY_SAMPLES,
                )
                ax.plot(
                    y_iv,
                    z_iv,
                    linestyle="--",
                    linewidth=1.1,
                    color=line.get_color(),
                    label=f"SC{sc_index + 1} IV",
                )

                centre = boresight_centre_on_plane(apex, boresight, x_plane)
                if centre is not None:
                    ax.scatter(centre[0], centre[1], marker="x", s=28)

            ax.scatter(0.0, 0.0, marker="+", s=50, label="SECR x-axis")
            ax.set_xlabel("SECR y at cross section (km)")
            ax.set_ylabel("SECR z at cross section (km)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            ax.set_title(
                f"{result.mode}: sampled epoch {epoch_index}, "
                f"x={x_plane:,.0f} km"
            )
            fig.tight_layout()

            filename = output_dir / (
                f"projection_{result.mode}_epoch-{epoch_index:04d}_"
                f"range-{distance_km:.0f}km.svg"
            )
            fig.savefig(filename, format="svg", bbox_inches="tight")
            saved.append(filename)
            if show:
                plt.show()
            else:
                plt.close(fig)

    return saved


def save_summary_figures(
    output_dir: Path,
    geometry: StudyGeometry,
    results: list[PackingResult],
    show: bool,
) -> list[Path]:
    saved: list[Path] = []
    x = np.arange(len(geometry.times))

    for result in results:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for sc_index in range(result.offsets_rad.shape[1]):
            ax.plot(
                x,
                np.rad2deg(result.offsets_rad[:, sc_index]),
                label=f"SC{sc_index + 1}",
            )
        ax.set_xlabel("Sampled LPF epoch index")
        ax.set_ylabel("Signed boresight offset from nominal (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"Boresight offsets: {result.mode}")
        fig.tight_layout()
        path = output_dir / f"boresight_offsets_{result.mode}.svg"
        fig.savefig(path, format="svg", bbox_inches="tight")
        saved.append(path)
        if show:
            plt.show()
        else:
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for sc_index in range(result.iv_clearance_rad.shape[1]):
            ax.plot(
                x,
                np.rad2deg(result.iv_clearance_rad[:, sc_index]),
                label=f"SC{sc_index + 1}",
            )
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Sampled LPF epoch index")
        ax.set_ylabel("Full-FOV IV clearance (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"Dynamic IV clearance: {result.mode}")
        fig.tight_layout()
        path = output_dir / f"iv_clearance_{result.mode}.svg"
        fig.savefig(path, format="svg", bbox_inches="tight")
        saved.append(path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    dynamic = next((r for r in results if r.mode == "dynamic_per_spacecraft"), None)
    if dynamic is not None:
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        ax.step(x, dynamic.inward_count, where="mid")
        ax.set_xlabel("Sampled LPF epoch index")
        ax.set_ylabel("Number of inward offset spacecraft")
        ax.grid(True, alpha=0.3)
        ax.set_title("Epochwise inward packing capacity")
        fig.tight_layout()
        path = output_dir / "dynamic_inward_count.svg"
        fig.savefig(path, format="svg", bbox_inches="tight")
        saved.append(path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved


# =============================================================================
# Reporting
# =============================================================================


def save_summary_csv(
    output_dir: Path,
    geometry: StudyGeometry,
    results: list[PackingResult],
) -> Path:
    rows: list[dict] = []
    epochs, num_spacecraft, _ = geometry.spacecraft_positions_secr_km.shape

    for result in results:
        for epoch_index in range(epochs):
            row: dict[str, object] = {
                "mode": result.mode,
                "sample_epoch_index": epoch_index,
                "source_period_index": int(geometry.source_indices[epoch_index]),
                "time": str(geometry.times[epoch_index]),
                "inward_count": int(result.inward_count[epoch_index]),
                "minimum_iv_clearance_deg": float(
                    np.rad2deg(np.min(result.iv_clearance_rad[epoch_index]))
                ),
                "minimum_pairwise_boresight_separation_deg": float(
                    np.rad2deg(
                        minimum_pairwise_separation_rad(
                            result.boresights_secr[epoch_index]
                        )
                    )
                ),
                "nominal_violation_count": int(
                    np.count_nonzero(result.nominal_clearance_rad[epoch_index] < 0.0)
                ),
                "largest_required_outward_shift_magnitude_deg": float(
                    abs(np.rad2deg(np.min(result.minimum_outward_shift_rad[epoch_index])))
                ),
            }
            for sc_index in range(num_spacecraft):
                row[f"sc{sc_index + 1}_offset_deg"] = float(
                    np.rad2deg(result.offsets_rad[epoch_index, sc_index])
                )
                row[f"sc{sc_index + 1}_iv_clearance_deg"] = float(
                    np.rad2deg(result.iv_clearance_rad[epoch_index, sc_index])
                )
                row[f"sc{sc_index + 1}_nominal_iv_clearance_deg"] = float(
                    np.rad2deg(result.nominal_clearance_rad[epoch_index, sc_index])
                )
                row[f"sc{sc_index + 1}_minimum_outward_shift_deg"] = float(
                    np.rad2deg(result.minimum_outward_shift_rad[epoch_index, sc_index])
                )
                for component, name in enumerate(("x", "y", "z")):
                    row[f"sc{sc_index + 1}_tangent_{name}"] = float(
                        result.tangent_secr[epoch_index, sc_index, component]
                    )
                    row[f"sc{sc_index + 1}_boresight_{name}"] = float(
                        result.boresights_secr[epoch_index, sc_index, component]
                    )
            rows.append(row)

    path = output_dir / "boresight_packing_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_study_metadata(
    output_dir: Path,
    config_path: Path,
    config: dict,
    geometry: StudyGeometry,
    fov_half_angle_rad: float,
    separation_rad: float,
    protected_half_angle_rad: float,
) -> Path:
    metadata = {
        "config_path": str(config_path),
        "orbit_file_path": str(_resolve_relative_path(config["orbit_file_path"], config_path)),
        "num_spacecraft": int(geometry.spacecraft_positions_secr_km.shape[1]),
        "requested_phase_deg": geometry.requested_phase_deg.tolist(),
        "realized_phase_deg": geometry.realized_phase_deg.tolist(),
        "phase_offset_indices": geometry.phase_offset_indices.tolist(),
        "nominal_boresight_secr": _unit(
            NOMINAL_BORESIGHT_SECR, "nominal"
        ).tolist(),
        "fov_area_deg2": float(config["fov"]),
        "fov_half_angle_deg": float(np.rad2deg(fov_half_angle_rad)),
        "fov_full_angular_diameter_deg": float(np.rad2deg(2.0 * fov_half_angle_rad)),
        "fov_separation_margin_deg": float(FOV_SEPARATION_MARGIN_DEG),
        "boresight_separation_deg": float(np.rad2deg(separation_rad)),
        "nominal_violation_policy": str(NOMINAL_VIOLATION_POLICY),
        "outward_clearance_buffer_deg": float(OUTWARD_CLEARANCE_BUFFER_DEG),
        "packing_search_step_deg": float(PACKING_SEARCH_STEP_DEG),
        "max_outward_offset_deg": float(MAX_OUTWARD_OFFSET_DEG),
        "iv_half_angle_deg": float(IV_HALF_ANGLE_DEG),
        "iv_clearance_margin_deg": float(IV_CLEARANCE_MARGIN_DEG),
        "protected_iv_centre_separation_deg": float(
            np.rad2deg(protected_half_angle_rad)
        ),
        "modes_to_run": list(MODES_TO_RUN),
        "fixed_tangent_source": str(FIXED_TANGENT_SOURCE),
        "design_sample_stride": int(DESIGN_SAMPLE_STRIDE),
        "three_d_epoch_selection_mode": str(THREE_D_EPOCH_SELECTION_MODE),
        "three_d_epoch_values": list(THREE_D_EPOCH_VALUES),
        "three_d_cone_length_km": float(THREE_D_CONE_LENGTH_KM),
        "three_d_show_iv_cones": bool(THREE_D_SHOW_IV_CONES),
        "three_d_show_earth_moon_los": bool(THREE_D_SHOW_EARTH_MOON_LOS),
        "sample_count": int(len(geometry.times)),
        "snapshot_fractions": list(SNAPSHOT_FRACTIONS),
        "cross_section_distances_km": list(CROSS_SECTION_DISTANCES_KM),
    }
    path = output_dir / "study_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed and per-spacecraft epochwise boresight packing."
    )
    parser.add_argument("config", help="Mission YAML configuration path.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening interactive windows.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_configuration(config_path)
    geometry = load_lpf_study_geometry(config_path, config)

    nominal = _unit(NOMINAL_BORESIGHT_SECR, "NOMINAL_BORESIGHT_SECR")
    fov_half_angle_rad = fov_area_deg2_to_half_angle_rad(float(config["fov"]))
    separation_rad = (
        2.0 * fov_half_angle_rad
        + np.deg2rad(float(FOV_SEPARATION_MARGIN_DEG))
    )
    protected_half_angle_rad = (
        geometry.iv_half_angle_rad
        + fov_half_angle_rad
        + np.deg2rad(float(IV_CLEARANCE_MARGIN_DEG))
    )

    requested_modes = tuple(str(mode).strip().lower() for mode in MODES_TO_RUN)
    allowed_modes = {"fixed_global", "dynamic_per_spacecraft"}
    unknown_modes = sorted(set(requested_modes) - allowed_modes)
    if unknown_modes:
        raise ValueError(
            f"Unsupported MODES_TO_RUN entries: {unknown_modes}. "
            f"Allowed values are {sorted(allowed_modes)}."
        )
    if not requested_modes:
        raise ValueError("MODES_TO_RUN must contain at least one packing mode.")

    results: list[PackingResult] = []
    fixed: PackingResult | None = None
    dynamic: PackingResult | None = None

    if "fixed_global" in requested_modes:
        try:
            fixed = solve_fixed_global(
                geometry,
                nominal,
                separation_rad,
                protected_half_angle_rad,
            )
            results.append(fixed)
        except RuntimeError as exc:
            print(f"WARNING: fixed_global solution unavailable: {exc}")

    if "dynamic_per_spacecraft" in requested_modes:
        dynamic = solve_dynamic_per_spacecraft(
            geometry,
            nominal,
            separation_rad,
            protected_half_angle_rad,
        )
        results.append(dynamic)

    if not results:
        raise RuntimeError("None of the requested packing modes produced a solution.")

    output_dir = Path(OUTPUT_FOLDER).expanduser()
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    show = SHOW_FIGURES_BY_DEFAULT and not args.no_show

    summary_csv = save_summary_csv(output_dir, geometry, results)
    metadata_path = save_study_metadata(
        output_dir,
        config_path,
        config,
        geometry,
        fov_half_angle_rad,
        separation_rad,
        protected_half_angle_rad,
    )
    figure_paths = save_summary_figures(output_dir, geometry, results, show)
    for result in results:
        figure_paths.extend(
            save_projection_figures(
                output_dir,
                geometry,
                result,
                fov_half_angle_rad,
                show,
            )
        )
        figure_paths.extend(
            save_3d_orbit_cone_figures(
                output_dir,
                geometry,
                result,
                fov_half_angle_rad,
                show,
            )
        )

    print("Initial-boresight packing study complete.")
    print(f"  LPF sampled epochs: {len(geometry.times)}")
    print(f"  Spacecraft: {geometry.spacecraft_positions_secr_km.shape[1]}")
    print(f"  FOV half-angle: {np.rad2deg(fov_half_angle_rad):.6f} deg")
    print(f"  Required boresight spacing: {np.rad2deg(separation_rad):.6f} deg")
    print(
        "  Required IV-axis-to-boresight-centre separation: "
        f"{np.rad2deg(protected_half_angle_rad):.6f} deg"
    )
    if fixed is not None:
        print(
            "  Fixed pattern inward count: "
            f"{int(fixed.inward_count[0])} of "
            f"{geometry.spacecraft_positions_secr_km.shape[1] - 1} offset spacecraft"
        )
        print(
            "  Fixed SC1 offset from nominal: "
            f"{np.rad2deg(fixed.offsets_rad[0, 0]):.6f} deg"
        )
    if dynamic is not None:
        print(
            "  Dynamic inward-count range: "
            f"{int(np.min(dynamic.inward_count))} to "
            f"{int(np.max(dynamic.inward_count))}"
        )
    if fixed is not None:
        print(
            "  Minimum fixed IV clearance: "
            f"{np.rad2deg(np.min(fixed.iv_clearance_rad)):.6f} deg"
        )
    if dynamic is not None:
        print(
            "  Dynamic epochs with at least one nominal violation: "
            f"{int(np.count_nonzero(np.any(dynamic.nominal_clearance_rad < 0.0, axis=1)))}"
        )
        print(
            "  Largest minimum outward correction: "
            f"{abs(np.rad2deg(np.min(dynamic.minimum_outward_shift_rad))):.6f} deg"
        )
        print(
            "  Minimum dynamic IV clearance: "
            f"{np.rad2deg(np.min(dynamic.iv_clearance_rad)):.6f} deg"
        )
    print(f"  Summary CSV: {summary_csv}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Figures written: {len(figure_paths)}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()