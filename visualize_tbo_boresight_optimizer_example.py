from __future__ import annotations

"""
Visualize the simplified single-epoch TBO boresight-optimization example.

This script uses the current candidate-dependent SNR optimizer and produces:

1. An angular boresight-score heatmap:
       azimuth about the Earth--Moon IV-zone axis
       versus off-axis angle from the IV-zone axis.
2. Three orthographic maximum-SNR projections for the selected fixed
   boresight: x-y, x-z, and y-z.
3. Three orthographic residence-time projections with the selected boresight,
   projected FOV, projected IV zone, and SNR-threshold boundary overlaid.
4. A JSON summary of the selected solution and adopted payload scenario.

The spatial plots are finite-volume orthographic projections:

- residence time is summed through the omitted Cartesian axis;
- SNR is reduced using the maximum value through the omitted axis;
- the FOV and IV-zone masks are projected using an ``any`` reduction through
  the omitted axis.

This makes all spacecraft, Earth, Moon, FOV, IV-zone, residence, and SNR
quantities refer to the same finite three-dimensional plotting volume.

Required files in the same Python environment
---------------------------------------------
search_tbo_boresight_optimizer_direct_spice_updated.py
payload_reference_scenario.py
payload_asteroid_snr_model.py
earth_moon_invisibility_zone_direct_spice.py
utilities.py
de430.bsp
naif0012.tls

Run directly in PyCharm. Set the working directory to the directory containing
those files and the SPICE kernels.
"""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter
import spiceypy as sp

import payload_reference_scenario as reference_payload
import search_tbo_boresight_optimizer_direct_spice_updated as optimizer


FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


# =============================================================================
# CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
OUTPUT_DIRECTORY = Path("boresight_visualization_output")
OUTPUT_STEM = "simplified_single_epoch"
SAVE_PNG = True
SAVE_SVG = True
SAVE_PDF = False
FIGURE_DPI = 300
SHOW_FIGURES = True

# -----------------------------------------------------------------------------
# Single-epoch simplified example
# -----------------------------------------------------------------------------
JD_TDB = 2451545.0
OBSERVER_DISTANCE_TOWARD_SUN_KM = 1.50e6

SNR_THRESHOLD = 5.0
FOV_HALF_ANGLE_DEG = 6.0
MAXIMUM_BORESIGHT_CHANGE_DEG = 15.0
CANDIDATE_ANGULAR_SPACING_DEG = 1.5
IV_ZONE_MARGIN_DEG = 1.0

# The visualization runs a full optimization. The shortcut is not needed for
# a single plotted epoch.
ENABLE_PREVIOUS_SCORE_SHORTCUT = False
SHORTCUT_MINIMUM_PREVIOUS_SCORE_FRACTION = 0.95

SNR_BATCH_SIZE = 2_000
CANDIDATE_BATCH_SIZE = 64
LOS_BATCH_SIZE = 2_000

# -----------------------------------------------------------------------------
# Angular score map
# -----------------------------------------------------------------------------
ANGLE_MAP_AZIMUTH_MIN_DEG = -180.0
ANGLE_MAP_AZIMUTH_MAX_DEG = 180.0
ANGLE_MAP_AZIMUTH_SAMPLES = 181
ANGLE_MAP_OFF_AXIS_MIN_DEG = 0.0
ANGLE_MAP_OFF_AXIS_MAX_DEG = 55.0
ANGLE_MAP_OFF_AXIS_SAMPLES = 111
ANGLE_MAP_SCORE_CONTOURS = 8

# -----------------------------------------------------------------------------
# Spatial SNR / residence projections
# -----------------------------------------------------------------------------
# The automatic volume contains the residence cells, spacecraft, Earth, and
# Moon. Increase the padding if cone boundaries are clipped.
SCENE_PADDING_FRACTION = 0.10
SCENE_MINIMUM_SPAN_KM = np.array([2.2e6, 1.4e6, 1.4e6], dtype=float)
GRID_SHAPE = (61, 51, 51)

# Residence points are deposited into a 3D weighted histogram before being
# projected. Smoothing is only for visualization; it does not affect the
# optimization score.
RESIDENCE_GAUSSIAN_SIGMA_VOXELS = 1.15

# SNR values are evaluated only inside the selected FOV and outside physical
# body interiors. The displayed SNR projection is the maximum through the
# omitted axis.
MINIMUM_OBSERVER_RANGE_KM = 1.0
SNR_COLOR_SCALE: Literal["linear", "log"] = "linear"
SNR_PLOT_MIN: float | None = 0.0
SNR_PLOT_MAX: float | None = None
SNR_FILLED_LEVELS = 40

RESIDENCE_COLOR_SCALE: Literal["linear", "log"] = "linear"
RESIDENCE_FILLED_LEVELS = 40

# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------
PLOT_DISTANCE_SCALE_KM = 1.0e6
PLOT_DISTANCE_UNIT_LABEL = r"$10^6$ km"

IV_ZONE_ALPHA = 0.18
IV_ZONE_HATCH = "//"
FOV_LINEWIDTH = 1.4
BORESIGHT_LINEWIDTH = 1.8
SNR_THRESHOLD_LINEWIDTH = 1.5
BODY_OUTLINE_LINEWIDTH = 1.1
SPACECRAFT_MARKER_SIZE = 95.0
ANNOTATE_SCENE_OBJECTS = True


# =============================================================================
# Basic utilities
# =============================================================================


def unit_vector(vector: ArrayLike, name: str) -> FloatArray:
    """Return a validated three-component unit vector."""

    values = np.asarray(vector, dtype=float)
    if values.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return np.asarray(values / norm, dtype=float)


def orthogonal_basis(axis_unit: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return a deterministic right-handed basis normal to ``axis_unit``."""

    axis = unit_vector(axis_unit, "axis_unit")
    helpers = np.eye(3)
    helper = helpers[int(np.argmin(np.abs(helpers @ axis)))]
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return np.asarray(first), np.asarray(second)


def direction_from_axis_angles(
    axis_unit: ArrayLike,
    off_axis_angle_rad: ArrayLike,
    azimuth_rad: ArrayLike,
) -> FloatArray:
    """Construct directions at angular coordinates about an arbitrary axis."""

    axis = unit_vector(axis_unit, "axis_unit")
    basis_1, basis_2 = orthogonal_basis(axis)

    off_axis = np.asarray(off_axis_angle_rad, dtype=float)
    azimuth = np.asarray(azimuth_rad, dtype=float)
    off_axis, azimuth = np.broadcast_arrays(off_axis, azimuth)

    directions = (
        np.cos(off_axis)[..., None] * axis
        + np.sin(off_axis)[..., None]
        * (
            np.cos(azimuth)[..., None] * basis_1
            + np.sin(azimuth)[..., None] * basis_2
        )
    )
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    return np.asarray(directions / norms, dtype=float)


def axis_angles_from_direction(
    direction: ArrayLike,
    axis_unit: ArrayLike,
) -> tuple[float, float]:
    """Return off-axis and azimuth angles about ``axis_unit`` in radians."""

    vector = unit_vector(direction, "direction")
    axis = unit_vector(axis_unit, "axis_unit")
    basis_1, basis_2 = orthogonal_basis(axis)

    off_axis = float(np.arccos(np.clip(vector @ axis, -1.0, 1.0)))
    azimuth = float(
        np.arctan2(vector @ basis_2, vector @ basis_1)
    )
    return off_axis, azimuth


def centre_axis_to_edges(centres: FloatArray) -> FloatArray:
    """Return histogram edges for a monotonically increasing centre axis."""

    values = np.asarray(centres, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("centres must be a one-dimensional axis of length >= 2.")
    differences = np.diff(values)
    if np.any(differences <= 0.0):
        raise ValueError("centres must be strictly increasing.")

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * differences[0]
    edges[-1] = values[-1] + 0.5 * differences[-1]
    return edges


def json_safe(value):
    """Convert nested NumPy/dataclass values to JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def validate_configuration() -> None:
    """Validate user-editable configuration values."""

    if ANGLE_MAP_AZIMUTH_SAMPLES < 2:
        raise ValueError("ANGLE_MAP_AZIMUTH_SAMPLES must be >= 2.")
    if ANGLE_MAP_OFF_AXIS_SAMPLES < 2:
        raise ValueError("ANGLE_MAP_OFF_AXIS_SAMPLES must be >= 2.")
    if len(GRID_SHAPE) != 3 or any(int(value) < 3 for value in GRID_SHAPE):
        raise ValueError("GRID_SHAPE must contain three integers >= 3.")
    if np.asarray(SCENE_MINIMUM_SPAN_KM).shape != (3,):
        raise ValueError("SCENE_MINIMUM_SPAN_KM must have shape (3,).")
    if np.any(np.asarray(SCENE_MINIMUM_SPAN_KM) <= 0.0):
        raise ValueError("SCENE_MINIMUM_SPAN_KM must be positive.")
    if SCENE_PADDING_FRACTION < 0.0:
        raise ValueError("SCENE_PADDING_FRACTION must be nonnegative.")
    if SNR_COLOR_SCALE not in {"linear", "log"}:
        raise ValueError("SNR_COLOR_SCALE must be 'linear' or 'log'.")
    if RESIDENCE_COLOR_SCALE not in {"linear", "log"}:
        raise ValueError("RESIDENCE_COLOR_SCALE must be 'linear' or 'log'.")


# =============================================================================
# Simplified example setup and optimization
# =============================================================================


def build_simplified_example() -> dict[str, object]:
    """Build and run the same physical simplified example as the optimizer."""

    optimizer.ivz.furnish_spice_kernels()
    et = optimizer.ivz.jd_tdb_to_et(JD_TDB)
    sun_position_geo_eme_km, _ = sp.spkpos(
        "SUN",
        et,
        "J2000",
        "NONE",
        "EARTH",
    )
    sun_direction_geo_eme = unit_vector(
        sun_position_geo_eme_km,
        "sun_direction_geo_eme",
    )
    observer_position_geo_eme_km = (
        OBSERVER_DISTANCE_TOWARD_SUN_KM * sun_direction_geo_eme
    )

    geometry = optimizer.build_single_epoch_search_geometry(
        jd_tdb=JD_TDB,
        observer_position_geo_eme_km=observer_position_geo_eme_km,
        observer_velocity_geo_eme_km_s=np.zeros(3, dtype=float),
    )

    iv_axis = geometry.invisibility_zone_axis_synodic
    initial_boresight_synodic = direction_from_axis_angles(
        iv_axis,
        np.deg2rad(25.0),
        0.0,
    )
    initial_boresight_eme = unit_vector(
        optimizer.geo_secr_to_geo_eme(
            initial_boresight_synodic,
            geometry.frame_context,
        ),
        "initial_boresight_eme",
    )

    config = optimizer.PointingConfig(
        snr_threshold=SNR_THRESHOLD,
        fov_half_angle_rad=np.deg2rad(FOV_HALF_ANGLE_DEG),
        maximum_boresight_change_rad=np.deg2rad(
            MAXIMUM_BORESIGHT_CHANGE_DEG
        ),
        candidate_angular_spacing_rad=np.deg2rad(
            CANDIDATE_ANGULAR_SPACING_DEG
        ),
        invisibility_zone_margin_rad=np.deg2rad(IV_ZONE_MARGIN_DEG),
        enable_previous_score_shortcut=ENABLE_PREVIOUS_SCORE_SHORTCUT,
        shortcut_minimum_previous_score_fraction=(
            SHORTCUT_MINIMUM_PREVIOUS_SCORE_FRACTION
        ),
        snr_batch_size=SNR_BATCH_SIZE,
        candidate_batch_size=CANDIDATE_BATCH_SIZE,
        los_batch_size=LOS_BATCH_SIZE,
    )

    scenario = reference_payload.make_reference_payload_scenario()
    snr_evaluator = optimizer.PayloadSNRGridEvaluator.from_reference_scenario(
        geometry=geometry,
        scenario=scenario,
    )


    residence_path = Path( "tbo_residence_time_results_2/xyz_synthetic_residence_grid_sparse.csv")
    optimizer._make_demo_residence_csv(
        residence_path,
        observer_position_synodic_km=(
            geometry.observer_position_synodic_km
        ),
        invisibility_zone_axis_synodic=iv_axis,
    )
    residence_grid = optimizer.load_sparse_residence_grid_csv(
        residence_path
    )

    result = optimizer.determine_search_boresight(
        residence_grid=residence_grid,
        observer_position_synodic_km=(
            geometry.observer_position_synodic_km
        ),
        previous_boresight_eme=initial_boresight_eme,
        previous_residence_score_days=None,
        invisibility_zone_axis_synodic=iv_axis,
        invisibility_zone_half_angle_rad=(
            geometry.invisibility_zone_half_angle_rad
        ),
        frame_context=geometry.frame_context,
        snr_evaluator=snr_evaluator,
        config=config,
    )

    if result.status != optimizer.PointingStatus.SUCCESS_OPTIMIZED_BORESIGHT:
        raise RuntimeError(
            "The simplified example did not produce an optimized boresight. "
            f"Status: {result.status.value}"
        )
    if result.boresight_synodic is None:
        raise RuntimeError("The optimizer returned no SECR boresight.")

    return {
        "geometry": geometry,
        "config": config,
        "scenario": scenario,
        "snr_evaluator": snr_evaluator,
        "residence_grid": residence_grid,
        "initial_boresight_synodic": initial_boresight_synodic,
        "initial_boresight_eme": initial_boresight_eme,
        "result": result,
    }


# =============================================================================
# Angular score map
# =============================================================================


def evaluate_angular_score_map(
    residence_grid: optimizer.ResidenceGrid,
    geometry: optimizer.SingleEpochSearchGeometry,
    initial_boresight_synodic: FloatArray,
    snr_evaluator: optimizer.SNREvaluator,
    config: optimizer.PointingConfig,
) -> dict[str, FloatArray | BoolArray]:
    """Evaluate a regular angular score grid about the IV-zone axis."""

    azimuth_deg = np.linspace(
        ANGLE_MAP_AZIMUTH_MIN_DEG,
        ANGLE_MAP_AZIMUTH_MAX_DEG,
        ANGLE_MAP_AZIMUTH_SAMPLES,
    )
    off_axis_deg = np.linspace(
        ANGLE_MAP_OFF_AXIS_MIN_DEG,
        ANGLE_MAP_OFF_AXIS_MAX_DEG,
        ANGLE_MAP_OFF_AXIS_SAMPLES,
    )
    off_axis_mesh_deg, azimuth_mesh_deg = np.meshgrid(
        off_axis_deg,
        azimuth_deg,
        indexing="ij",
    )

    candidates = direction_from_axis_angles(
        geometry.invisibility_zone_axis_synodic,
        np.deg2rad(off_axis_mesh_deg),
        np.deg2rad(azimuth_mesh_deg),
    ).reshape(-1, 3)

    slew_feasible = (
        candidates @ unit_vector(
            initial_boresight_synodic,
            "initial_boresight_synodic",
        )
        >= np.cos(config.maximum_boresight_change_rad) - 1.0e-15
    )
    iv_feasible = optimizer.invisibility_zone_feasible_mask(
        candidate_boresights_synodic=candidates,
        invisibility_zone_axis_synodic=(
            geometry.invisibility_zone_axis_synodic
        ),
        invisibility_zone_half_angle_rad=(
            geometry.invisibility_zone_half_angle_rad
        ),
        fov_half_angle_rad=config.fov_half_angle_rad,
        invisibility_zone_margin_rad=(
            config.invisibility_zone_margin_rad
        ),
    )
    feasible = slew_feasible & iv_feasible

    positions = np.asarray(
        residence_grid.positions_synodic_km,
        dtype=float,
    )
    residence = np.asarray(
        residence_grid.residence_time_days,
        dtype=float,
    )
    los, _, valid = optimizer.observer_relative_los(
        positions,
        geometry.observer_position_synodic_km,
    )
    positions = positions[valid]
    residence = residence[valid]
    los = los[valid]

    scores_flat = np.full(candidates.shape[0], np.nan, dtype=float)
    geometric_flat = np.zeros(candidates.shape[0], dtype=np.int64)
    detectable_flat = np.zeros(candidates.shape[0], dtype=np.int64)

    feasible_indices = np.flatnonzero(feasible)
    if feasible_indices.size:
        scores, geometric, detectable = (
            optimizer.score_candidates_with_candidate_dependent_snr(
                candidate_boresights=candidates[feasible_indices],
                positions_synodic_km=positions,
                los_unit_vectors=los,
                residence_days=residence,
                snr_evaluator=snr_evaluator,
                config=config,
            )
        )
        scores_flat[feasible_indices] = scores
        geometric_flat[feasible_indices] = geometric
        detectable_flat[feasible_indices] = detectable

    shape = off_axis_mesh_deg.shape
    return {
        "azimuth_deg": azimuth_deg,
        "off_axis_deg": off_axis_deg,
        "scores_days": scores_flat.reshape(shape),
        "geometric_counts": geometric_flat.reshape(shape),
        "detectable_counts": detectable_flat.reshape(shape),
        "slew_feasible": slew_feasible.reshape(shape),
        "iv_feasible": iv_feasible.reshape(shape),
        "feasible": feasible.reshape(shape),
    }


def plot_angular_score_map(
    angular_map: dict[str, FloatArray | BoolArray],
    geometry: optimizer.SingleEpochSearchGeometry,
    initial_boresight_synodic: FloatArray,
    selected_boresight_synodic: FloatArray,
    config: optimizer.PointingConfig,
) -> None:
    """Plot angular score, hard IV-zone exclusion, and slew feasibility."""

    azimuth_deg = np.asarray(angular_map["azimuth_deg"], dtype=float)
    off_axis_deg = np.asarray(angular_map["off_axis_deg"], dtype=float)
    scores = np.asarray(angular_map["scores_days"], dtype=float)
    slew_feasible = np.asarray(
        angular_map["slew_feasible"],
        dtype=bool,
    )
    iv_feasible = np.asarray(angular_map["iv_feasible"], dtype=bool)

    azimuth_mesh, off_axis_mesh = np.meshgrid(
        azimuth_deg,
        off_axis_deg,
        indexing="xy",
    )

    figure, axis = plt.subplots(figsize=(7.6, 5.8))
    masked_scores = np.ma.masked_invalid(scores)
    finite_scores = scores[np.isfinite(scores)]

    if finite_scores.size:
        image = axis.pcolormesh(
            azimuth_mesh,
            off_axis_mesh,
            masked_scores,
            shading="auto",
        )
        colourbar = figure.colorbar(image, ax=axis)
        colourbar.set_label("Detectable residence score (days)")

        minimum = float(np.min(finite_scores))
        maximum = float(np.max(finite_scores))
        if maximum > minimum and ANGLE_MAP_SCORE_CONTOURS >= 2:
            levels = np.linspace(
                minimum,
                maximum,
                ANGLE_MAP_SCORE_CONTOURS,
            )[1:-1]
            if levels.size:
                lines = axis.contour(
                    azimuth_mesh,
                    off_axis_mesh,
                    masked_scores,
                    levels=levels,
                    linewidths=0.7,
                )
                axis.clabel(lines, inline=True, fontsize=7, fmt="%.0f")
    else:
        axis.text(
            0.5,
            0.5,
            "No feasible angular samples",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )

    # The IV constraint is axisymmetric in these coordinates.
    required_axis_separation_deg = np.rad2deg(
        geometry.invisibility_zone_half_angle_rad
        + config.fov_half_angle_rad
        + config.invisibility_zone_margin_rad
    )
    axis.axhspan(
        off_axis_deg[0],
        min(required_axis_separation_deg, off_axis_deg[-1]),
        alpha=IV_ZONE_ALPHA,
        hatch=IV_ZONE_HATCH,
        label="IV-zone/FOV forbidden",
    )

    # Gray out directions outside the maximum-change cone.
    outside_slew = (~slew_feasible).astype(float)
    if np.any(outside_slew > 0.5):
        axis.contourf(
            azimuth_mesh,
            off_axis_mesh,
            outside_slew,
            levels=[0.5, 1.5],
            alpha=0.18,
        )

    # Draw the exact sampled IV-feasibility transition as a consistency check.
    if np.any(iv_feasible) and np.any(~iv_feasible):
        axis.contour(
            azimuth_mesh,
            off_axis_mesh,
            iv_feasible.astype(float),
            levels=[0.5],
            linewidths=1.0,
            linestyles="--",
        )

    initial_off_axis, initial_azimuth = axis_angles_from_direction(
        initial_boresight_synodic,
        geometry.invisibility_zone_axis_synodic,
    )
    selected_off_axis, selected_azimuth = axis_angles_from_direction(
        selected_boresight_synodic,
        geometry.invisibility_zone_axis_synodic,
    )

    axis.scatter(
        [np.rad2deg(initial_azimuth)],
        [np.rad2deg(initial_off_axis)],
        marker="o",
        s=55,
        facecolors="none",
        linewidths=1.4,
        label="Reference boresight",
        zorder=6,
    )
    axis.scatter(
        [np.rad2deg(selected_azimuth)],
        [np.rad2deg(selected_off_axis)],
        marker="*",
        s=125,
        label="Selected boresight",
        zorder=7,
    )

    axis.set_xlim(azimuth_deg[0], azimuth_deg[-1])
    axis.set_ylim(off_axis_deg[0], off_axis_deg[-1])
    axis.set_xlabel("Azimuth about IV-zone axis (deg)")
    axis.set_ylabel("Off-axis angle from IV-zone axis (deg)")
    axis.set_title("Boresight score map")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    save_figure(figure, "boresight_score_heatmap")


# =============================================================================
# Three-dimensional spatial fields
# =============================================================================


def make_scene_axes(
    residence_positions_secr_km: FloatArray,
    observer_position_secr_km: FloatArray,
    earth_position_secr_km: FloatArray,
    moon_position_secr_km: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build automatic Cartesian axes containing the complete example scene."""

    points = np.vstack(
        [
            np.asarray(residence_positions_secr_km, dtype=float),
            np.asarray(observer_position_secr_km, dtype=float).reshape(1, 3),
            np.asarray(earth_position_secr_km, dtype=float).reshape(1, 3),
            np.asarray(moon_position_secr_km, dtype=float).reshape(1, 3),
        ]
    )
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    centre = 0.5 * (minimum + maximum)
    span = maximum - minimum
    span = np.maximum(span, np.asarray(SCENE_MINIMUM_SPAN_KM, dtype=float))
    span *= 1.0 + 2.0 * SCENE_PADDING_FRACTION

    axes = []
    for component, samples in zip(span, GRID_SHAPE):
        half = 0.5 * component
        axis_index = len(axes)
        axes.append(
            np.linspace(
                centre[axis_index] - half,
                centre[axis_index] + half,
                int(samples),
            )
        )
    return tuple(np.asarray(axis, dtype=float) for axis in axes)  # type: ignore[return-value]


def flat_grid_positions(
    x_km: FloatArray,
    y_km: FloatArray,
    z_km: FloatArray,
) -> FloatArray:
    """Return all Cartesian grid centres as rows."""

    x_grid, y_grid, z_grid = np.meshgrid(
        x_km,
        y_km,
        z_km,
        indexing="ij",
    )
    return np.column_stack(
        (x_grid.ravel(), y_grid.ravel(), z_grid.ravel())
    )


def build_spatial_fields(
    residence_grid: optimizer.ResidenceGrid,
    geometry: optimizer.SingleEpochSearchGeometry,
    selected_boresight_synodic: FloatArray,
    snr_evaluator: optimizer.SNREvaluator,
    config: optimizer.PointingConfig,
) -> dict[str, object]:
    """Evaluate 3D SNR, residence, FOV, and IV-zone fields."""

    observer = np.asarray(
        geometry.observer_position_synodic_km,
        dtype=float,
    )
    earth = np.asarray(
        optimizer.geo_eme_to_geo_secr(
            geometry.earth_position_geo_eme_km,
            geometry.frame_context,
        ),
        dtype=float,
    )
    moon = np.asarray(
        optimizer.geo_eme_to_geo_secr(
            geometry.moon_position_geo_eme_km,
            geometry.frame_context,
        ),
        dtype=float,
    )
    sun = np.asarray(
        optimizer.geo_eme_to_geo_secr(
            geometry.sun_position_geo_eme_km,
            geometry.frame_context,
        ),
        dtype=float,
    )

    x_km, y_km, z_km = make_scene_axes(
        residence_grid.positions_synodic_km,
        observer,
        earth,
        moon,
    )
    positions = flat_grid_positions(x_km, y_km, z_km)
    shape = (x_km.size, y_km.size, z_km.size)

    relative = positions - observer[None, :]
    ranges = np.linalg.norm(relative, axis=1)
    valid_los = np.isfinite(ranges) & (ranges > MINIMUM_OBSERVER_RANGE_KM)
    los = np.zeros_like(relative)
    los[valid_los] = relative[valid_los] / ranges[valid_los, None]

    selected = unit_vector(
        selected_boresight_synodic,
        "selected_boresight_synodic",
    )
    iv_axis = unit_vector(
        geometry.invisibility_zone_axis_synodic,
        "invisibility_zone_axis_synodic",
    )

    fov_mask_flat = (
        valid_los
        & (
            los @ selected
            >= np.cos(config.fov_half_angle_rad)
        )
    )
    iv_zone_mask_flat = (
        valid_los
        & (
            los @ iv_axis
            >= np.cos(geometry.invisibility_zone_half_angle_rad)
        )
    )

    scenario = reference_payload.make_reference_payload_scenario()
    earth_radius = float(scenario.environment.earth.radius_km)
    moon_radius = float(scenario.environment.moon.radius_km)
    sun_radius = float(scenario.environment.solar_radius_km)

    outside_bodies = (
        np.linalg.norm(positions - earth[None, :], axis=1) > earth_radius
    )
    outside_bodies &= (
        np.linalg.norm(positions - moon[None, :], axis=1) > moon_radius
    )
    outside_bodies &= (
        np.linalg.norm(positions - sun[None, :], axis=1) > sun_radius
    )

    snr_flat = np.full(positions.shape[0], np.nan, dtype=float)
    evaluate_mask = fov_mask_flat & outside_bodies
    evaluate_indices = np.flatnonzero(evaluate_mask)
    if evaluate_indices.size:
        snr_flat[evaluate_indices] = optimizer.evaluate_snr_in_batches(
            positions_synodic_km=positions[evaluate_indices],
            boresights_synodic=selected,
            snr_evaluator=snr_evaluator,
            batch_size=config.snr_batch_size,
        )

    x_edges = centre_axis_to_edges(x_km)
    y_edges = centre_axis_to_edges(y_km)
    z_edges = centre_axis_to_edges(z_km)
    residence_3d, _ = np.histogramdd(
        np.asarray(residence_grid.positions_synodic_km, dtype=float),
        bins=(x_edges, y_edges, z_edges),
        weights=np.asarray(residence_grid.residence_time_days, dtype=float),
    )
    if RESIDENCE_GAUSSIAN_SIGMA_VOXELS > 0.0:
        residence_3d = gaussian_filter(
            residence_3d,
            sigma=RESIDENCE_GAUSSIAN_SIGMA_VOXELS,
            mode="constant",
        )

    return {
        "x_km": x_km,
        "y_km": y_km,
        "z_km": z_km,
        "snr_3d": snr_flat.reshape(shape),
        "residence_3d": residence_3d,
        "fov_mask_3d": fov_mask_flat.reshape(shape),
        "iv_zone_mask_3d": iv_zone_mask_flat.reshape(shape),
        "observer_secr_km": observer,
        "earth_secr_km": earth,
        "moon_secr_km": moon,
        "selected_boresight_synodic": selected,
        "earth_radius_km": earth_radius,
        "moon_radius_km": moon_radius,
    }


# =============================================================================
# Orthographic projections and scene overlays
# =============================================================================


PROJECTIONS = {
    "xy": {
        "horizontal_axis": 0,
        "vertical_axis": 1,
        "omit_axis": 2,
        "horizontal_label": "SECR x",
        "vertical_label": "SECR y",
    },
    "xz": {
        "horizontal_axis": 0,
        "vertical_axis": 2,
        "omit_axis": 1,
        "horizontal_label": "SECR x",
        "vertical_label": "SECR z",
    },
    "yz": {
        "horizontal_axis": 1,
        "vertical_axis": 2,
        "omit_axis": 0,
        "horizontal_label": "SECR y",
        "vertical_label": "SECR z",
    },
}


def project_nanmax(values_3d: FloatArray, omit_axis: int) -> FloatArray:
    """NaN-safe maximum projection without all-NaN warnings."""

    values = np.asarray(values_3d, dtype=float)
    finite = np.isfinite(values)
    replaced = np.where(finite, values, -np.inf)
    projected = np.max(replaced, axis=omit_axis)
    projected[~np.any(finite, axis=omit_axis)] = np.nan
    return np.asarray(projected, dtype=float)


def project_field(
    values_3d: FloatArray,
    omit_axis: int,
    reduction: Literal["sum", "max", "any"],
) -> FloatArray | BoolArray:
    """Project a 3D field through one omitted Cartesian axis."""

    if reduction == "sum":
        return np.asarray(np.nansum(values_3d, axis=omit_axis), dtype=float)
    if reduction == "max":
        return project_nanmax(values_3d, omit_axis)
    if reduction == "any":
        return np.asarray(np.any(values_3d, axis=omit_axis), dtype=bool)
    raise ValueError(f"Unsupported reduction: {reduction}")


def projected_scene_coordinates(
    position_km: ArrayLike,
    horizontal_axis: int,
    vertical_axis: int,
) -> tuple[float, float]:
    """Return scaled coordinates for one orthographic projection."""

    position = np.asarray(position_km, dtype=float)
    return (
        float(position[horizontal_axis] / PLOT_DISTANCE_SCALE_KM),
        float(position[vertical_axis] / PLOT_DISTANCE_SCALE_KM),
    )


def line_to_axes_boundary(
    origin_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip a positive ray to the current rectangular plotting boundary."""

    x0, y0 = origin_xy
    dx, dy = direction_xy
    norm = float(np.hypot(dx, dy))
    if norm <= 1.0e-14:
        return None
    dx /= norm
    dy /= norm

    candidates: list[float] = []
    if dx > 0.0:
        candidates.append((x_limits[1] - x0) / dx)
    elif dx < 0.0:
        candidates.append((x_limits[0] - x0) / dx)
    if dy > 0.0:
        candidates.append((y_limits[1] - y0) / dy)
    elif dy < 0.0:
        candidates.append((y_limits[0] - y0) / dy)

    positive = [value for value in candidates if value > 0.0]
    if not positive:
        return None
    distance = min(positive)
    return (x0, y0), (x0 + distance * dx, y0 + distance * dy)


def add_scene_overlays(
    axis: plt.Axes,
    horizontal_values_scaled: FloatArray,
    vertical_values_scaled: FloatArray,
    fov_projection: BoolArray,
    iv_projection: BoolArray,
    fields: dict[str, object],
    horizontal_axis: int,
    vertical_axis: int,
) -> None:
    """Add projected IV zone, FOV, boresight, bodies, and spacecraft."""

    horizontal_mesh, vertical_mesh = np.meshgrid(
        horizontal_values_scaled,
        vertical_values_scaled,
        indexing="ij",
    )

    if np.any(iv_projection):
        axis.contourf(
            horizontal_mesh,
            vertical_mesh,
            iv_projection.astype(float),
            levels=[0.5, 1.5],
            alpha=IV_ZONE_ALPHA,
            hatches=[IV_ZONE_HATCH],
            zorder=2,
        )

    if np.any(fov_projection) and np.any(~fov_projection):
        axis.contour(
            horizontal_mesh,
            vertical_mesh,
            fov_projection.astype(float),
            levels=[0.5],
            linewidths=FOV_LINEWIDTH,
            linestyles="--",
            zorder=6,
        )

    observer = np.asarray(fields["observer_secr_km"], dtype=float)
    earth = np.asarray(fields["earth_secr_km"], dtype=float)
    moon = np.asarray(fields["moon_secr_km"], dtype=float)
    boresight = np.asarray(
        fields["selected_boresight_synodic"],
        dtype=float,
    )

    observer_xy = projected_scene_coordinates(
        observer,
        horizontal_axis,
        vertical_axis,
    )
    boresight_direction_xy = (
        float(boresight[horizontal_axis]),
        float(boresight[vertical_axis]),
    )
    ray = line_to_axes_boundary(
        observer_xy,
        boresight_direction_xy,
        (float(horizontal_values_scaled[0]), float(horizontal_values_scaled[-1])),
        (float(vertical_values_scaled[0]), float(vertical_values_scaled[-1])),
    )
    if ray is not None:
        (x0, y0), (x1, y1) = ray
        axis.plot(
            [x0, x1],
            [y0, y1],
            linewidth=BORESIGHT_LINEWIDTH,
            zorder=7,
            label="Selected boresight",
        )

    body_specs = [
        (
            "Earth",
            earth,
            float(fields["earth_radius_km"]),
            "o",
        ),
        (
            "Moon",
            moon,
            float(fields["moon_radius_km"]),
            "o",
        ),
    ]
    for name, position, radius_km, marker in body_specs:
        x_body, y_body = projected_scene_coordinates(
            position,
            horizontal_axis,
            vertical_axis,
        )
        axis.add_patch(
            Circle(
                (x_body, y_body),
                radius_km / PLOT_DISTANCE_SCALE_KM,
                facecolor="none",
                edgecolor="black",
                linewidth=BODY_OUTLINE_LINEWIDTH,
                zorder=8,
            )
        )
        axis.scatter(
            [x_body],
            [y_body],
            marker=marker,
            s=20,
            edgecolors="black",
            facecolors="white",
            linewidths=0.8,
            zorder=9,
        )
        if ANNOTATE_SCENE_OBJECTS:
            axis.annotate(
                name,
                xy=(x_body, y_body),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                zorder=10,
            )

    axis.scatter(
        [observer_xy[0]],
        [observer_xy[1]],
        marker="*",
        s=SPACECRAFT_MARKER_SIZE,
        edgecolors="black",
        facecolors="white",
        linewidths=0.9,
        zorder=11,
    )
    if ANNOTATE_SCENE_OBJECTS:
        axis.annotate(
            "Spacecraft",
            xy=observer_xy,
            xytext=(6, -8),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="top",
            zorder=12,
        )


def choose_snr_limits(values: FloatArray) -> tuple[float, float]:
    """Choose robust filled-contour limits for projected SNR."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if SNR_COLOR_SCALE == "log":
        finite = finite[finite > 0.0]
    if finite.size == 0:
        raise ValueError("The projected SNR field has no finite values.")

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    vmin = data_min if SNR_PLOT_MIN is None else float(SNR_PLOT_MIN)
    vmax = data_max if SNR_PLOT_MAX is None else float(SNR_PLOT_MAX)

    if SNR_COLOR_SCALE == "log":
        vmin = max(vmin, np.finfo(float).tiny)
    if vmax <= vmin:
        delta = max(abs(vmin) * 0.01, 1.0e-9)
        if SNR_COLOR_SCALE == "log":
            vmax = vmin * 1.01
        else:
            vmin -= delta
            vmax += delta
    return vmin, vmax


def plot_snr_projection(
    projection_name: str,
    fields: dict[str, object],
    config: optimizer.PointingConfig,
) -> None:
    """Plot one orthographic maximum-SNR projection."""

    specification = PROJECTIONS[projection_name]
    horizontal_axis = int(specification["horizontal_axis"])
    vertical_axis = int(specification["vertical_axis"])
    omit_axis = int(specification["omit_axis"])

    axes = [
        np.asarray(fields["x_km"], dtype=float),
        np.asarray(fields["y_km"], dtype=float),
        np.asarray(fields["z_km"], dtype=float),
    ]
    horizontal = axes[horizontal_axis] / PLOT_DISTANCE_SCALE_KM
    vertical = axes[vertical_axis] / PLOT_DISTANCE_SCALE_KM
    horizontal_mesh, vertical_mesh = np.meshgrid(
        horizontal,
        vertical,
        indexing="ij",
    )

    snr_projection = np.asarray(
        project_field(
            np.asarray(fields["snr_3d"], dtype=float),
            omit_axis,
            "max",
        ),
        dtype=float,
    )
    fov_projection = np.asarray(
        project_field(
            np.asarray(fields["fov_mask_3d"], dtype=bool),
            omit_axis,
            "any",
        ),
        dtype=bool,
    )
    iv_projection = np.asarray(
        project_field(
            np.asarray(fields["iv_zone_mask_3d"], dtype=bool),
            omit_axis,
            "any",
        ),
        dtype=bool,
    )

    vmin, vmax = choose_snr_limits(snr_projection)
    if SNR_COLOR_SCALE == "log":
        levels = np.geomspace(vmin, vmax, SNR_FILLED_LEVELS)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        levels = np.linspace(vmin, vmax, SNR_FILLED_LEVELS)
        norm = Normalize(vmin=vmin, vmax=vmax)

    figure, axis = plt.subplots(figsize=(7.0, 5.7))
    masked = np.ma.masked_invalid(snr_projection)
    filled = axis.contourf(
        horizontal_mesh,
        vertical_mesh,
        masked,
        levels=levels,
        norm=norm,
        extend="max",
        zorder=1,
    )
    colourbar = figure.colorbar(filled, ax=axis)
    colourbar.set_label("Maximum projected SNR")

    finite = snr_projection[np.isfinite(snr_projection)]
    if (
        finite.size
        and float(np.min(finite)) <= config.snr_threshold
        <= float(np.max(finite))
    ):
        threshold_line = axis.contour(
            horizontal_mesh,
            vertical_mesh,
            masked,
            levels=[config.snr_threshold],
            linewidths=SNR_THRESHOLD_LINEWIDTH,
            zorder=5,
        )
        axis.clabel(
            threshold_line,
            inline=True,
            fontsize=8,
            fmt={config.snr_threshold: f"SNR = {config.snr_threshold:g}"},
        )

    add_scene_overlays(
        axis=axis,
        horizontal_values_scaled=horizontal,
        vertical_values_scaled=vertical,
        fov_projection=fov_projection,
        iv_projection=iv_projection,
        fields=fields,
        horizontal_axis=horizontal_axis,
        vertical_axis=vertical_axis,
    )

    axis.set_xlabel(
        f"{specification['horizontal_label']} ({PLOT_DISTANCE_UNIT_LABEL})"
    )
    axis.set_ylabel(
        f"{specification['vertical_label']} ({PLOT_DISTANCE_UNIT_LABEL})"
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"Selected-boresight SNR projection ({projection_name})"
    )
    figure.tight_layout()
    save_figure(figure, f"selected_boresight_snr_{projection_name}")


def plot_residence_projection(
    projection_name: str,
    fields: dict[str, object],
    config: optimizer.PointingConfig,
) -> None:
    """Plot residence time with the projected SNR-threshold boundary."""

    specification = PROJECTIONS[projection_name]
    horizontal_axis = int(specification["horizontal_axis"])
    vertical_axis = int(specification["vertical_axis"])
    omit_axis = int(specification["omit_axis"])

    axes = [
        np.asarray(fields["x_km"], dtype=float),
        np.asarray(fields["y_km"], dtype=float),
        np.asarray(fields["z_km"], dtype=float),
    ]
    horizontal = axes[horizontal_axis] / PLOT_DISTANCE_SCALE_KM
    vertical = axes[vertical_axis] / PLOT_DISTANCE_SCALE_KM
    horizontal_mesh, vertical_mesh = np.meshgrid(
        horizontal,
        vertical,
        indexing="ij",
    )

    residence_projection = np.asarray(
        project_field(
            np.asarray(fields["residence_3d"], dtype=float),
            omit_axis,
            "sum",
        ),
        dtype=float,
    )
    snr_projection = np.asarray(
        project_field(
            np.asarray(fields["snr_3d"], dtype=float),
            omit_axis,
            "max",
        ),
        dtype=float,
    )
    fov_projection = np.asarray(
        project_field(
            np.asarray(fields["fov_mask_3d"], dtype=bool),
            omit_axis,
            "any",
        ),
        dtype=bool,
    )
    iv_projection = np.asarray(
        project_field(
            np.asarray(fields["iv_zone_mask_3d"], dtype=bool),
            omit_axis,
            "any",
        ),
        dtype=bool,
    )

    positive = residence_projection[residence_projection > 0.0]
    if positive.size == 0:
        raise ValueError("The projected residence field is empty.")

    if RESIDENCE_COLOR_SCALE == "log":
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
        levels = np.geomspace(vmin, vmax, RESIDENCE_FILLED_LEVELS)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        residence_for_plot = np.ma.masked_less_equal(
            residence_projection,
            0.0,
        )
    else:
        vmin = 0.0
        vmax = float(np.max(positive))
        levels = np.linspace(vmin, vmax, RESIDENCE_FILLED_LEVELS)
        norm = Normalize(vmin=vmin, vmax=vmax)
        residence_for_plot = np.ma.masked_less_equal(
            residence_projection,
            0.0,
        )

    figure, axis = plt.subplots(figsize=(7.0, 5.7))
    filled = axis.contourf(
        horizontal_mesh,
        vertical_mesh,
        residence_for_plot,
        levels=levels,
        norm=norm,
        extend="max",
        zorder=1,
    )
    colourbar = figure.colorbar(filled, ax=axis)
    colourbar.set_label("Projected residence time (days)")

    finite_snr = snr_projection[np.isfinite(snr_projection)]
    if (
        finite_snr.size
        and float(np.min(finite_snr)) <= config.snr_threshold
        <= float(np.max(finite_snr))
    ):
        threshold_line = axis.contour(
            horizontal_mesh,
            vertical_mesh,
            np.ma.masked_invalid(snr_projection),
            levels=[config.snr_threshold],
            linewidths=SNR_THRESHOLD_LINEWIDTH,
            zorder=5,
        )
        axis.clabel(
            threshold_line,
            inline=True,
            fontsize=8,
            fmt={config.snr_threshold: f"SNR = {config.snr_threshold:g}"},
        )

    add_scene_overlays(
        axis=axis,
        horizontal_values_scaled=horizontal,
        vertical_values_scaled=vertical,
        fov_projection=fov_projection,
        iv_projection=iv_projection,
        fields=fields,
        horizontal_axis=horizontal_axis,
        vertical_axis=vertical_axis,
    )

    axis.set_xlabel(
        f"{specification['horizontal_label']} ({PLOT_DISTANCE_UNIT_LABEL})"
    )
    axis.set_ylabel(
        f"{specification['vertical_label']} ({PLOT_DISTANCE_UNIT_LABEL})"
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"Residence distribution and SNR threshold ({projection_name})"
    )
    figure.tight_layout()
    save_figure(figure, f"residence_with_snr_threshold_{projection_name}")


# =============================================================================
# Output and main workflow
# =============================================================================


def save_figure(figure: plt.Figure, filename_stem: str) -> None:
    """Save and optionally show one completed Matplotlib figure."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        figure.savefig(
            OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_{filename_stem}.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    if SAVE_SVG:
        figure.savefig(
            OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_{filename_stem}.svg",
            bbox_inches="tight",
        )
    if SAVE_PDF:
        figure.savefig(
            OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_{filename_stem}.pdf",
            bbox_inches="tight",
        )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(figure)


def save_summary(example: dict[str, object]) -> None:
    """Save the adopted scenario and selected optimization result."""

    geometry = example["geometry"]
    config = example["config"]
    scenario = example["scenario"]
    result = example["result"]

    assert isinstance(geometry, optimizer.SingleEpochSearchGeometry)
    assert isinstance(config, optimizer.PointingConfig)
    assert isinstance(
        scenario,
        reference_payload.ReferencePayloadScenario,
    )
    assert isinstance(result, optimizer.PointingResult)

    payload = {
        "jd_tdb": geometry.jd_tdb,
        "observer_position_secr_km": geometry.observer_position_synodic_km,
        "observer_position_geo_eme_km": (
            geometry.observer_position_geo_eme_km
        ),
        "iv_zone_axis_secr": geometry.invisibility_zone_axis_synodic,
        "iv_zone_half_angle_deg": np.rad2deg(
            geometry.invisibility_zone_half_angle_rad
        ),
        "earth_moon_angular_separation_deg": np.rad2deg(
            geometry.invisibility_zone.earth_moon_angular_separation_rad
        ),
        "pointing_config": asdict(config),
        "reference_payload": asdict(scenario.payload),
        "reference_asteroid": asdict(scenario.asteroid),
        "asteroid_velocity_geo_eme_km_s": (
            scenario.asteroid_velocity_geo_eme_km_s
        ),
        "asteroid_angular_rate_override_arcsec_s": (
            scenario.asteroid_angular_rate_override_arcsec_s
        ),
        "selected_status": result.status.value,
        "selected_boresight_secr": result.boresight_synodic,
        "selected_boresight_geo_eme": result.boresight_inertial,
        "selected_score_days": result.residence_score_days,
        "selected_geometric_cells": result.number_of_cells_in_selected_fov,
        "selected_detectable_cells": result.number_of_detectable_cells,
        "candidate_count": result.number_of_candidate_boresights,
        "feasible_candidate_count": result.number_of_feasible_boresights,
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (
        OUTPUT_DIRECTORY
        / f"{OUTPUT_STEM}_visualization_summary.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(json_safe(payload), stream, indent=2)


def main() -> None:
    """Run the optimizer and generate the complete visualization package."""

    validate_configuration()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Building and optimizing the simplified single-epoch example...")
    example = build_simplified_example()

    geometry = example["geometry"]
    config = example["config"]
    scenario = example["scenario"]
    snr_evaluator = example["snr_evaluator"]
    residence_grid = example["residence_grid"]
    initial_boresight_synodic = example["initial_boresight_synodic"]
    result = example["result"]

    assert isinstance(geometry, optimizer.SingleEpochSearchGeometry)
    assert isinstance(config, optimizer.PointingConfig)
    assert isinstance(
        scenario,
        reference_payload.ReferencePayloadScenario,
    )
    assert isinstance(snr_evaluator, optimizer.PayloadSNRGridEvaluator)
    assert isinstance(residence_grid, optimizer.ResidenceGrid)
    assert isinstance(result, optimizer.PointingResult)
    assert result.boresight_synodic is not None

    print("Evaluating angular score map...")
    angular_map = evaluate_angular_score_map(
        residence_grid=residence_grid,
        geometry=geometry,
        initial_boresight_synodic=np.asarray(
            initial_boresight_synodic,
            dtype=float,
        ),
        snr_evaluator=snr_evaluator,
        config=config,
    )
    plot_angular_score_map(
        angular_map=angular_map,
        geometry=geometry,
        initial_boresight_synodic=np.asarray(
            initial_boresight_synodic,
            dtype=float,
        ),
        selected_boresight_synodic=result.boresight_synodic,
        config=config,
    )

    print("Evaluating projected spatial SNR and residence fields...")
    fields = build_spatial_fields(
        residence_grid=residence_grid,
        geometry=geometry,
        selected_boresight_synodic=result.boresight_synodic,
        snr_evaluator=snr_evaluator,
        config=config,
    )

    for projection_name in PROJECTIONS:
        print(f"Plotting {projection_name} SNR projection...")
        plot_snr_projection(
            projection_name=projection_name,
            fields=fields,
            config=config,
        )
        print(f"Plotting {projection_name} residence projection...")
        plot_residence_projection(
            projection_name=projection_name,
            fields=fields,
            config=config,
        )

    save_summary(example)

    print("Visualization workflow completed successfully.")
    print(f"Output directory: {OUTPUT_DIRECTORY.resolve()}")
    print(f"Selected score: {result.residence_score_days:.3f} days")
    print(
        "Selected boresight (SECR): "
        f"{np.array2string(result.boresight_synodic, precision=6)}"
    )
    print(
        "Detected cells in selected FOV: "
        f"{result.number_of_detectable_cells} / "
        f"{result.number_of_cells_in_selected_fov}"
    )
    print(
        "Reference asteroid: "
        f"H={reference_payload.ASTEROID_ABSOLUTE_MAGNITUDE:.1f}, "
        f"p_V={reference_payload.ASTEROID_GEOMETRIC_ALBEDO:.3f}, "
        f"G12={reference_payload.ASTEROID_G12:.3f}, "
        f"speed="
        f"{reference_payload.ASTEROID_APPARENT_SPEED_ARCSEC_PER_HOUR:.1f} "
        "arcsec/hr"
    )


if __name__ == "__main__":
    main()
