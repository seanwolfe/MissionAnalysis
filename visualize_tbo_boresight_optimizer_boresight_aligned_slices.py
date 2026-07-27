from __future__ import annotations

"""
Visualize one single-epoch TBO search-boresight optimization using an existing
sparse three-dimensional residence-time CSV.

Outputs
-------
1. Angular score heatmap with:
       x = off-axis angle from the Earth--Moon IV-zone axis
       y = azimuth about the IV-zone axis
       colour = detectable residence-time score

2. Three boresight-aligned residence/SNR views:
       - boresight--IV longitudinal plane;
       - orthogonal boresight longitudinal plane;
       - transverse cross-boresight plane at a residence-weighted range.

Each spatial view contains a finite-thickness residence-time slab, selected-
boresight SNR contour lines, the FOV boundary, the IV-zone cross-section, and
projected spacecraft/Earth/Moon geometry. The first longitudinal plane is
constructed to contain both the selected boresight and IV-zone axes, so both
cones are always visible. The cross-boresight plane is placed automatically at
the residence-weighted median forward range of cells inside the selected FOV.

The input residence CSV is never created or overwritten by this script.

Required files/modules
----------------------
search_tbo_boresight_optimizer_direct_spice_updated.py
payload_reference_scenario.py
payload_asteroid_snr_model.py
earth_moon_invisibility_zone_direct_spice.py
utilities.py
de430.bsp
naif0012.tls
"""

from dataclasses import asdict
import json
from pathlib import Path
from typing import TypeAlias

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

AU_KM = 149_597_870.7


# =============================================================================
# CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Input residence grid
# -----------------------------------------------------------------------------
# A relative path is checked first from the current working directory and then
# relative to this script's directory. An absolute path is also accepted.
RESIDENCE_GRID_CSV = Path(
    r"tbo_residence_time_results_2/xyz_synthetic_residence_grid_sparse_real.csv"
)

# Expected CSV columns are those used by the optimizer loader:
#   Synodic x (km)
#   Synodic y (km)
#   Synodic z (km)
#   residence_time_days

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
OUTPUT_DIRECTORY = Path("boresight_visualization_output")
OUTPUT_STEM = "single_epoch_actual_residence"
SAVE_PNG = True
SAVE_SVG = True
SAVE_PDF = False
FIGURE_DPI = 300
SHOW_FIGURES = True

# -----------------------------------------------------------------------------
# Single epoch and optimizer
# -----------------------------------------------------------------------------
JD_TDB = 2451545.0
OBSERVER_DISTANCE_TOWARD_SUN_KM = 1.50e6

SNR_THRESHOLD = 1.0
FOV_HALF_ANGLE_DEG = 6.0
MAXIMUM_BORESIGHT_CHANGE_DEG = 45.0
CANDIDATE_ANGULAR_SPACING_DEG = 1.5
IV_ZONE_MARGIN_DEG = 1.0

ENABLE_PREVIOUS_SCORE_SHORTCUT = False
SHORTCUT_MINIMUM_PREVIOUS_SCORE_FRACTION = 0.95

SNR_BATCH_SIZE = 200_000
CANDIDATE_BATCH_SIZE = 2048
LOS_BATCH_SIZE = 200_000

# Initial direction relative to the IV-zone axis.
INITIAL_BORESIGHT_OFF_AXIS_DEG = 25.0
INITIAL_BORESIGHT_AZIMUTH_DEG = 0.0

# -----------------------------------------------------------------------------
# Angular score map
# -----------------------------------------------------------------------------
ANGLE_MAP_OFF_AXIS_MIN_DEG = 0.0
ANGLE_MAP_OFF_AXIS_MAX_DEG = 55.0
ANGLE_MAP_OFF_AXIS_SAMPLES = 55
ANGLE_MAP_AZIMUTH_MIN_DEG = -180.0
ANGLE_MAP_AZIMUTH_MAX_DEG = 180.0
ANGLE_MAP_AZIMUTH_SAMPLES = 180
ANGLE_MAP_SCORE_CONTOURS = 8

# -----------------------------------------------------------------------------
# Boresight-aligned spatial slices
# -----------------------------------------------------------------------------
# The original residence grid is defined within approximately +/-0.01 AU in
# the Earth-centred SECR frame. The new plots use local coordinates attached to
# the selected boresight rather than fixed x/y/z planes.
SPATIAL_HALF_WIDTH_AU = 0.01

# Number of samples in each evaluated 2D SNR plane. Runtime scales with the
# product. A value of 161 gives 25,921 SNR points per plane.
PLANE_GRID_SHAPE = (161, 161)

# Longitudinal planes are centred on the spacecraft. The horizontal coordinate
# is distance along the selected boresight; the vertical coordinate is one of
# the two transverse basis directions.
LONGITUDINAL_ALONG_MIN_AU = -0.002
LONGITUDINAL_ALONG_MAX_AU = 0.018
LONGITUDINAL_TRANSVERSE_HALF_WIDTH_AU = 0.010

# The cross-boresight plane is placed at the residence-weighted median forward
# range of cells inside the selected FOV. Its half-width is sized from the FOV
# footprint and limited by these bounds.
CROSS_PLANE_MIN_HALF_WIDTH_AU = 0.0005
CROSS_PLANE_MAX_HALF_WIDTH_AU = 0.010
CROSS_PLANE_FOV_PADDING_FACTOR = 1.35
CROSS_PLANE_FALLBACK_RANGE_AU = 0.005

# Residence time is accumulated from a finite slab around each plane. The slab
# grows automatically until enough sparse residence cells are captured or the
# maximum half-thickness is reached.
RESIDENCE_SLICE_INITIAL_HALF_THICKNESS_AU = 0.00010
RESIDENCE_SLICE_MAX_HALF_THICKNESS_AU = 0.00200
RESIDENCE_SLICE_THICKNESS_GROWTH = 1.6
MINIMUM_RESIDENCE_CELLS_IN_SLICE = 100

# Smoothing affects only the rendered 2D residence histogram, not optimization.
RESIDENCE_GAUSSIAN_SIGMA_BINS = 1.0
RESIDENCE_COLOR_SCALE = "log"  # "linear" or "log"
RESIDENCE_FILLED_LEVELS = 50

# SNR contours are calculated using the actual selected fixed boresight.
SNR_CONTOUR_LEVELS = (1.0, 2.0, 3.0, 4.0, 5.0)
MASK_SNR_CONTOURS_OUTSIDE_FOV = True
MINIMUM_OBSERVER_RANGE_KM = 1.0

# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------
PLOT_DISTANCE_SCALE_KM = AU_KM
PLOT_DISTANCE_UNIT_LABEL = "AU"

IV_ZONE_ALPHA = 0.18
IV_ZONE_HATCH = "//"
FOV_LINEWIDTH = 1.4
BORESIGHT_LINEWIDTH = 1.8
SNR_CONTOUR_LINEWIDTH = 1.15
BODY_OUTLINE_LINEWIDTH = 1.1
SPACECRAFT_MARKER_SIZE = 95.0
ANNOTATE_SCENE_OBJECTS = True


# =============================================================================
# Utilities
# =============================================================================


def resolve_input_path(path: Path) -> Path:
    """Resolve an input path without silently creating or replacing it."""

    candidate = path.expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Residence grid not found: {resolved}")
        return resolved

    working_directory_candidate = (Path.cwd() / candidate).resolve()
    if working_directory_candidate.exists():
        return working_directory_candidate

    script_directory_candidate = (
        Path(__file__).resolve().parent / candidate
    ).resolve()
    if script_directory_candidate.exists():
        return script_directory_candidate

    raise FileNotFoundError(
        "Residence grid not found. Checked:\n"
        f"  {working_directory_candidate}\n"
        f"  {script_directory_candidate}"
    )


def unit_vector(vector: ArrayLike, name: str) -> FloatArray:
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
    return np.asarray(
        directions / np.linalg.norm(directions, axis=-1, keepdims=True),
        dtype=float,
    )


def axis_angles_from_direction(
    direction: ArrayLike,
    axis_unit: ArrayLike,
) -> tuple[float, float]:
    vector = unit_vector(direction, "direction")
    axis = unit_vector(axis_unit, "axis_unit")
    basis_1, basis_2 = orthogonal_basis(axis)
    off_axis = float(np.arccos(np.clip(vector @ axis, -1.0, 1.0)))
    azimuth = float(np.arctan2(vector @ basis_2, vector @ basis_1))
    return off_axis, azimuth


def centre_axis_to_edges(centres: FloatArray) -> FloatArray:
    values = np.asarray(centres, dtype=float)
    differences = np.diff(values)
    if values.ndim != 1 or values.size < 2 or np.any(differences <= 0.0):
        raise ValueError("centres must be strictly increasing and length >= 2.")
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * differences[0]
    edges[-1] = values[-1] + 0.5 * differences[-1]
    return edges


def save_figure(figure: plt.Figure, filename_stem: str) -> None:
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


def json_safe(value):
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
    if len(PLANE_GRID_SHAPE) != 2 or any(
        int(value) < 3 for value in PLANE_GRID_SHAPE
    ):
        raise ValueError("PLANE_GRID_SHAPE must contain two integers >= 3.")
    if SPATIAL_HALF_WIDTH_AU <= 0.0:
        raise ValueError("SPATIAL_HALF_WIDTH_AU must be positive.")
    if LONGITUDINAL_ALONG_MAX_AU <= LONGITUDINAL_ALONG_MIN_AU:
        raise ValueError(
            "LONGITUDINAL_ALONG_MAX_AU must exceed "
            "LONGITUDINAL_ALONG_MIN_AU."
        )
    if LONGITUDINAL_TRANSVERSE_HALF_WIDTH_AU <= 0.0:
        raise ValueError(
            "LONGITUDINAL_TRANSVERSE_HALF_WIDTH_AU must be positive."
        )
    if not (
        0.0 < CROSS_PLANE_MIN_HALF_WIDTH_AU
        <= CROSS_PLANE_MAX_HALF_WIDTH_AU
    ):
        raise ValueError("Invalid cross-plane half-width bounds.")
    if CROSS_PLANE_FOV_PADDING_FACTOR <= 0.0:
        raise ValueError("CROSS_PLANE_FOV_PADDING_FACTOR must be positive.")
    if RESIDENCE_SLICE_INITIAL_HALF_THICKNESS_AU <= 0.0:
        raise ValueError(
            "RESIDENCE_SLICE_INITIAL_HALF_THICKNESS_AU must be positive."
        )
    if (
        RESIDENCE_SLICE_MAX_HALF_THICKNESS_AU
        < RESIDENCE_SLICE_INITIAL_HALF_THICKNESS_AU
    ):
        raise ValueError(
            "Maximum residence-slice half-thickness must not be smaller "
            "than the initial value."
        )
    if RESIDENCE_SLICE_THICKNESS_GROWTH <= 1.0:
        raise ValueError(
            "RESIDENCE_SLICE_THICKNESS_GROWTH must exceed 1."
        )
    if MINIMUM_RESIDENCE_CELLS_IN_SLICE < 1:
        raise ValueError(
            "MINIMUM_RESIDENCE_CELLS_IN_SLICE must be >= 1."
        )
    if RESIDENCE_COLOR_SCALE not in {"linear", "log"}:
        raise ValueError("RESIDENCE_COLOR_SCALE must be 'linear' or 'log'.")
    if ANGLE_MAP_OFF_AXIS_SAMPLES < 2 or ANGLE_MAP_AZIMUTH_SAMPLES < 2:
        raise ValueError("Angular-map sample counts must be >= 2.")
    if not SNR_CONTOUR_LEVELS:
        raise ValueError("SNR_CONTOUR_LEVELS must not be empty.")


# =============================================================================
# Single-epoch optimization
# =============================================================================


def build_single_epoch_example() -> dict[str, object]:
    """Load the configured residence CSV and run one full optimization."""

    residence_path = resolve_input_path(RESIDENCE_GRID_CSV)

    optimizer.ivz.furnish_spice_kernels()
    et = optimizer.ivz.jd_tdb_to_et(JD_TDB)
    sun_position_geo_eme_km, _ = sp.spkpos(
        "SUN", et, "J2000", "NONE", "EARTH"
    )
    sun_direction_geo_eme = unit_vector(
        sun_position_geo_eme_km,
        "sun_direction_geo_eme",
    )
    observer_position_geo_eme_km = (
        OBSERVER_DISTANCE_TOWARD_SUN_KM * sun_direction_geo_eme + [0, 0, 300_000]
    )

    geometry = optimizer.build_single_epoch_search_geometry(
        jd_tdb=JD_TDB,
        observer_position_geo_eme_km=observer_position_geo_eme_km,
        observer_velocity_geo_eme_km_s=np.zeros(3, dtype=float),
    )

    initial_boresight_synodic = direction_from_axis_angles(
        geometry.invisibility_zone_axis_synodic,
        np.deg2rad(INITIAL_BORESIGHT_OFF_AXIS_DEG),
        np.deg2rad(INITIAL_BORESIGHT_AZIMUTH_DEG),
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

    # The configured file is only read. No synthetic demo grid is generated.
    residence_grid = optimizer.load_sparse_residence_grid_csv(residence_path)

    result = optimizer.determine_search_boresight(
        residence_grid=residence_grid,
        observer_position_synodic_km=(
            geometry.observer_position_synodic_km
        ),
        previous_boresight_eme=initial_boresight_eme,
        previous_residence_score_days=None,
        invisibility_zone_axis_synodic=(
            geometry.invisibility_zone_axis_synodic
        ),
        invisibility_zone_half_angle_rad=(
            geometry.invisibility_zone_half_angle_rad
        ),
        frame_context=geometry.frame_context,
        snr_evaluator=snr_evaluator,
        config=config,
    )

    if result.status != optimizer.PointingStatus.SUCCESS_OPTIMIZED_BORESIGHT:
        raise RuntimeError(
            "The optimization did not return a selected boresight. "
            f"Status: {result.status.value}"
        )
    if result.boresight_synodic is None:
        raise RuntimeError("The optimizer returned no SECR boresight.")

    return {
        "residence_path": residence_path,
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
    """Evaluate score on a regular off-axis-angle/azimuth grid."""

    off_axis_deg = np.linspace(
        ANGLE_MAP_OFF_AXIS_MIN_DEG,
        ANGLE_MAP_OFF_AXIS_MAX_DEG,
        ANGLE_MAP_OFF_AXIS_SAMPLES,
    )
    azimuth_deg = np.linspace(
        ANGLE_MAP_AZIMUTH_MIN_DEG,
        ANGLE_MAP_AZIMUTH_MAX_DEG,
        ANGLE_MAP_AZIMUTH_SAMPLES,
    )
    off_axis_mesh_deg, azimuth_mesh_deg = np.meshgrid(
        off_axis_deg,
        azimuth_deg,
        indexing="xy",
    )

    candidates = direction_from_axis_angles(
        geometry.invisibility_zone_axis_synodic,
        np.deg2rad(off_axis_mesh_deg),
        np.deg2rad(azimuth_mesh_deg),
    ).reshape(-1, 3)

    initial = unit_vector(
        initial_boresight_synodic,
        "initial_boresight_synodic",
    )
    slew_feasible = (
        candidates @ initial
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
        invisibility_zone_margin_rad=config.invisibility_zone_margin_rad,
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

    finite_positive = (
            np.all(np.isfinite(positions), axis=1)
            & np.isfinite(residence)
            & (residence > 1.0))

    positions = positions[finite_positive]
    residence = residence[finite_positive]

    los, _, valid = optimizer.observer_relative_los(
        positions,
        geometry.observer_position_synodic_km,
    )
    positions = positions[valid]
    residence = residence[valid]
    los = los[valid]

    scores_flat = np.full(candidates.shape[0], np.nan, dtype=float)
    feasible_indices = np.flatnonzero(feasible)
    if feasible_indices.size:
        scores, _, _ = (
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

    shape = off_axis_mesh_deg.shape
    return {
        "off_axis_deg": off_axis_deg,
        "azimuth_deg": azimuth_deg,
        "scores_days": scores_flat.reshape(shape),
        "slew_feasible": slew_feasible.reshape(shape),
        "iv_feasible": iv_feasible.reshape(shape),
    }


def plot_angular_score_map(
    angular_map: dict[str, FloatArray | BoolArray],
    geometry: optimizer.SingleEpochSearchGeometry,
    initial_boresight_synodic: FloatArray,
    selected_boresight_synodic: FloatArray,
    config: optimizer.PointingConfig,
) -> None:
    """Plot off-axis angle on x and azimuth on y."""

    off_axis_deg = np.asarray(angular_map["off_axis_deg"], dtype=float)
    azimuth_deg = np.asarray(angular_map["azimuth_deg"], dtype=float)
    scores = np.asarray(angular_map["scores_days"], dtype=float)
    slew_feasible = np.asarray(angular_map["slew_feasible"], dtype=bool)
    iv_feasible = np.asarray(angular_map["iv_feasible"], dtype=bool)

    off_axis_mesh, azimuth_mesh = np.meshgrid(
        off_axis_deg,
        azimuth_deg,
        indexing="xy",
    )

    figure, axis = plt.subplots(figsize=(7.6, 5.8))
    masked_scores = np.ma.masked_invalid(scores)
    finite_scores = scores[np.isfinite(scores)]

    if finite_scores.size:
        image = axis.pcolormesh(
            off_axis_mesh,
            azimuth_mesh,
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
                    off_axis_mesh,
                    azimuth_mesh,
                    masked_scores,
                    levels=levels,
                    linewidths=0.7,
                )
                axis.clabel(lines, inline=True, fontsize=7, fmt="%.0f")

    required_axis_separation_deg = np.rad2deg(
        geometry.invisibility_zone_half_angle_rad
        + config.fov_half_angle_rad
        + config.invisibility_zone_margin_rad
    )
    axis.axvspan(
        off_axis_deg[0],
        min(required_axis_separation_deg, off_axis_deg[-1]),
        alpha=IV_ZONE_ALPHA,
        hatch=IV_ZONE_HATCH,
        label="IV-zone/FOV forbidden",
    )

    outside_slew = (~slew_feasible).astype(float)
    if np.any(outside_slew > 0.5):
        axis.contourf(
            off_axis_mesh,
            azimuth_mesh,
            outside_slew,
            levels=[0.5, 1.5],
            alpha=0.18,
        )

    if np.any(iv_feasible) and np.any(~iv_feasible):
        axis.contour(
            off_axis_mesh,
            azimuth_mesh,
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
        [np.rad2deg(initial_off_axis)],
        [np.rad2deg(initial_azimuth)],
        marker="o",
        s=55,
        facecolors="none",
        linewidths=1.4,
        label="Reference boresight",
        zorder=6,
    )
    axis.scatter(
        [np.rad2deg(selected_off_axis)],
        [np.rad2deg(selected_azimuth)],
        marker="*",
        s=125,
        label="Selected boresight",
        zorder=7,
    )

    axis.set_xlim(off_axis_deg[0], off_axis_deg[-1])
    axis.set_ylim(azimuth_deg[0], azimuth_deg[-1])
    axis.set_xlabel("Off-axis angle from IV-zone axis (deg)")
    axis.set_ylabel("Azimuth about IV-zone axis (deg)")
    axis.set_title("Boresight score map")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    save_figure(figure, "boresight_score_heatmap")


# =============================================================================
# Boresight-aligned residence slabs and SNR planes
# =============================================================================


def weighted_median(values: ArrayLike, weights: ArrayLike) -> float:
    values_array = np.asarray(values, dtype=float).reshape(-1)
    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    valid = (
        np.isfinite(values_array)
        & np.isfinite(weights_array)
        & (weights_array > 0.0)
    )
    if not np.any(valid):
        raise ValueError("weighted_median requires at least one positive weight.")
    values_array = values_array[valid]
    weights_array = weights_array[valid]
    order = np.argsort(values_array)
    values_array = values_array[order]
    weights_array = weights_array[order]
    cumulative = np.cumsum(weights_array)
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(values_array[min(index, values_array.size - 1)])


def make_boresight_aligned_basis(
    selected_boresight_synodic: ArrayLike,
    invisibility_zone_axis_synodic: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return boresight, in-plane IV, and orthogonal unit directions."""

    e1 = unit_vector(selected_boresight_synodic, "selected_boresight_synodic")
    iv_axis = unit_vector(
        invisibility_zone_axis_synodic,
        "invisibility_zone_axis_synodic",
    )

    iv_transverse = iv_axis - float(iv_axis @ e1) * e1
    norm = float(np.linalg.norm(iv_transverse))
    if norm <= 1.0e-10:
        # Deterministic fallback when the boresight and IV axis are nearly
        # parallel or antiparallel.
        e2, _ = orthogonal_basis(e1)
    else:
        e2 = iv_transverse / norm

    e3 = np.cross(e1, e2)
    e3 = unit_vector(e3, "boresight_orthogonal_axis")
    # Re-orthogonalize e2 to suppress roundoff.
    e2 = unit_vector(np.cross(e3, e1), "boresight_iv_plane_axis")
    return np.asarray(e1), np.asarray(e2), np.asarray(e3)


def select_cross_boresight_range_km(
    residence_grid: optimizer.ResidenceGrid,
    observer_position_synodic_km: ArrayLike,
    selected_boresight_synodic: ArrayLike,
    fov_half_angle_rad: float,
) -> float:
    """Choose a representative forward range for the transverse plane."""

    positions = np.asarray(residence_grid.positions_synodic_km, dtype=float)
    residence = np.asarray(residence_grid.residence_time_days, dtype=float)
    observer = np.asarray(observer_position_synodic_km, dtype=float)
    boresight = unit_vector(selected_boresight_synodic, "selected_boresight")

    relative = positions - observer[None, :]
    ranges = np.linalg.norm(relative, axis=1)
    valid = (
        np.all(np.isfinite(relative), axis=1)
        & np.isfinite(residence)
        & (residence > 0.0)
        & np.isfinite(ranges)
        & (ranges > 0.0)
    )
    los = np.zeros_like(relative)
    los[valid] = relative[valid] / ranges[valid, None]
    forward = relative @ boresight

    in_selected_fov = (
        valid
        & (forward > 0.0)
        & (los @ boresight >= np.cos(fov_half_angle_rad))
    )
    if np.any(in_selected_fov):
        return weighted_median(
            forward[in_selected_fov],
            residence[in_selected_fov],
        )

    forward_only = valid & (forward > 0.0)
    if np.any(forward_only):
        return weighted_median(
            forward[forward_only],
            residence[forward_only],
        )

    return float(CROSS_PLANE_FALLBACK_RANGE_AU * AU_KM)


def build_boresight_plane_definitions(
    residence_grid: optimizer.ResidenceGrid,
    geometry: optimizer.SingleEpochSearchGeometry,
    selected_boresight_synodic: FloatArray,
    config: optimizer.PointingConfig,
) -> dict[str, dict[str, object]]:
    """Construct the three visualization planes."""

    observer = np.asarray(geometry.observer_position_synodic_km, dtype=float)
    e1, e2, e3 = make_boresight_aligned_basis(
        selected_boresight_synodic,
        geometry.invisibility_zone_axis_synodic,
    )

    cross_range_km = select_cross_boresight_range_km(
        residence_grid=residence_grid,
        observer_position_synodic_km=observer,
        selected_boresight_synodic=e1,
        fov_half_angle_rad=config.fov_half_angle_rad,
    )
    fov_radius_au = (
        cross_range_km * np.tan(config.fov_half_angle_rad) / AU_KM
    )
    cross_half_width_au = float(
        np.clip(
            CROSS_PLANE_FOV_PADDING_FACTOR * fov_radius_au,
            CROSS_PLANE_MIN_HALF_WIDTH_AU,
            CROSS_PLANE_MAX_HALF_WIDTH_AU,
        )
    )

    return {
        "boresight_iv": {
            "title": "Boresight--IV longitudinal plane",
            "origin_km": observer,
            "u_axis": e1,
            "v_axis": e2,
            "normal_axis": e3,
            "u_limits_au": (
                LONGITUDINAL_ALONG_MIN_AU,
                LONGITUDINAL_ALONG_MAX_AU,
            ),
            "v_limits_au": (
                -LONGITUDINAL_TRANSVERSE_HALF_WIDTH_AU,
                LONGITUDINAL_TRANSVERSE_HALF_WIDTH_AU,
            ),
            "u_label": "Distance along selected boresight",
            "v_label": "Toward IV-zone axis in boresight plane",
            "view_type": "longitudinal",
            "cross_range_km": None,
        },
        "boresight_orthogonal": {
            "title": "Orthogonal boresight longitudinal plane",
            "origin_km": observer,
            "u_axis": e1,
            "v_axis": e3,
            "normal_axis": e2,
            "u_limits_au": (
                LONGITUDINAL_ALONG_MIN_AU,
                LONGITUDINAL_ALONG_MAX_AU,
            ),
            "v_limits_au": (
                -LONGITUDINAL_TRANSVERSE_HALF_WIDTH_AU,
                LONGITUDINAL_TRANSVERSE_HALF_WIDTH_AU,
            ),
            "u_label": "Distance along selected boresight",
            "v_label": "Orthogonal transverse distance",
            "view_type": "longitudinal",
            "cross_range_km": None,
        },
        "cross_boresight": {
            "title": (
                "Cross-boresight plane at residence-weighted range"
            ),
            "origin_km": observer + cross_range_km * e1,
            "u_axis": e2,
            "v_axis": e3,
            "normal_axis": e1,
            "u_limits_au": (-cross_half_width_au, cross_half_width_au),
            "v_limits_au": (-cross_half_width_au, cross_half_width_au),
            "u_label": "Transverse distance toward IV-zone axis",
            "v_label": "Orthogonal transverse distance",
            "view_type": "cross",
            "cross_range_km": cross_range_km,
        },
    }


def choose_residence_slab_mask(
    positions_synodic_km: FloatArray,
    plane_origin_km: FloatArray,
    plane_normal_unit: FloatArray,
) -> tuple[BoolArray, float]:
    """Adaptively choose a finite residence slab around one plane."""

    signed_distance_au = (
        (positions_synodic_km - plane_origin_km[None, :])
        @ plane_normal_unit
        / AU_KM
    )
    half_thickness = float(
        RESIDENCE_SLICE_INITIAL_HALF_THICKNESS_AU
    )
    maximum = float(RESIDENCE_SLICE_MAX_HALF_THICKNESS_AU)

    while True:
        mask = np.abs(signed_distance_au) <= half_thickness
        count = int(np.count_nonzero(mask))
        if count >= MINIMUM_RESIDENCE_CELLS_IN_SLICE:
            return np.asarray(mask, dtype=bool), half_thickness
        if half_thickness >= maximum:
            # A sparse grid can miss an arbitrarily thin geometric plane. To
            # keep the visualization informative, include the nearest cells
            # when the configured maximum slab remains under-populated.
            finite_indices = np.flatnonzero(np.isfinite(signed_distance_au))
            if finite_indices.size == 0:
                return np.asarray(mask, dtype=bool), half_thickness
            number_to_keep = min(
                MINIMUM_RESIDENCE_CELLS_IN_SLICE,
                finite_indices.size,
            )
            order = finite_indices[
                np.argsort(np.abs(signed_distance_au[finite_indices]))
            ]
            selected = order[:number_to_keep]
            fallback_mask = np.zeros(signed_distance_au.size, dtype=bool)
            fallback_mask[selected] = True
            effective_half_thickness = max(
                half_thickness,
                float(np.max(np.abs(signed_distance_au[selected]))),
            )
            return fallback_mask, effective_half_thickness
        half_thickness = min(
            maximum,
            half_thickness * RESIDENCE_SLICE_THICKNESS_GROWTH,
        )


def scene_states_secr(
    geometry: optimizer.SingleEpochSearchGeometry,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    observer = np.asarray(geometry.observer_position_synodic_km, dtype=float)
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
    return observer, earth, moon, sun


def evaluate_boresight_aligned_plane(
    plane_name: str,
    plane: dict[str, object],
    residence_grid: optimizer.ResidenceGrid,
    geometry: optimizer.SingleEpochSearchGeometry,
    selected_boresight_synodic: FloatArray,
    snr_evaluator: optimizer.SNREvaluator,
    config: optimizer.PointingConfig,
    scenario: reference_payload.ReferencePayloadScenario,
) -> dict[str, object]:
    """Evaluate residence and fixed-boresight SNR on one local plane."""

    origin = np.asarray(plane["origin_km"], dtype=float)
    u_axis = unit_vector(plane["u_axis"], "plane_u_axis")
    v_axis = unit_vector(plane["v_axis"], "plane_v_axis")
    normal = unit_vector(plane["normal_axis"], "plane_normal_axis")
    u_limits = tuple(float(value) for value in plane["u_limits_au"])
    v_limits = tuple(float(value) for value in plane["v_limits_au"])

    nu, nv = (int(value) for value in PLANE_GRID_SHAPE)
    u_edges_au = np.linspace(u_limits[0], u_limits[1], nu + 1)
    v_edges_au = np.linspace(v_limits[0], v_limits[1], nv + 1)
    u_centres_au = 0.5 * (u_edges_au[:-1] + u_edges_au[1:])
    v_centres_au = 0.5 * (v_edges_au[:-1] + v_edges_au[1:])
    u_mesh_au, v_mesh_au = np.meshgrid(
        u_centres_au,
        v_centres_au,
        indexing="ij",
    )

    positions = (
        origin[None, :]
        + (u_mesh_au.ravel() * AU_KM)[:, None] * u_axis[None, :]
        + (v_mesh_au.ravel() * AU_KM)[:, None] * v_axis[None, :]
    )

    residence_positions = np.asarray(
        residence_grid.positions_synodic_km,
        dtype=float,
    )
    residence_days = np.asarray(
        residence_grid.residence_time_days,
        dtype=float,
    )
    slab_mask, slab_half_thickness_au = choose_residence_slab_mask(
        residence_positions,
        origin,
        normal,
    )
    local_residence = residence_positions[slab_mask] - origin[None, :]
    residence_u_au = local_residence @ u_axis / AU_KM
    residence_v_au = local_residence @ v_axis / AU_KM
    residence_2d, _, _ = np.histogram2d(
        residence_u_au,
        residence_v_au,
        bins=(u_edges_au, v_edges_au),
        weights=residence_days[slab_mask],
    )
    if RESIDENCE_GAUSSIAN_SIGMA_BINS > 0.0:
        residence_2d = gaussian_filter(
            residence_2d,
            sigma=RESIDENCE_GAUSSIAN_SIGMA_BINS,
            mode="constant",
        )

    observer, earth, moon, sun = scene_states_secr(geometry)
    relative = positions - observer[None, :]
    ranges = np.linalg.norm(relative, axis=1)
    valid = np.isfinite(ranges) & (ranges > MINIMUM_OBSERVER_RANGE_KM)
    los = np.zeros_like(relative)
    los[valid] = relative[valid] / ranges[valid, None]

    selected = unit_vector(
        selected_boresight_synodic,
        "selected_boresight_synodic",
    )
    iv_axis = unit_vector(
        geometry.invisibility_zone_axis_synodic,
        "invisibility_zone_axis_synodic",
    )
    fov_mask = valid & (
        los @ selected >= np.cos(config.fov_half_angle_rad)
    )
    iv_mask = valid & (
        los @ iv_axis
        >= np.cos(geometry.invisibility_zone_half_angle_rad)
    )

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
    evaluate_indices = np.flatnonzero(valid & outside_bodies)
    if evaluate_indices.size:
        snr_flat[evaluate_indices] = optimizer.evaluate_snr_in_batches(
            positions_synodic_km=positions[evaluate_indices],
            boresights_synodic=selected,
            snr_evaluator=snr_evaluator,
            batch_size=config.snr_batch_size,
        )
    if MASK_SNR_CONTOURS_OUTSIDE_FOV:
        snr_flat[~fov_mask] = np.nan

    shape = u_mesh_au.shape
    finite_snr = snr_flat[np.isfinite(snr_flat)]
    print(
        f"{plane_name}: slab half-thickness = "
        f"{slab_half_thickness_au:.6f} AU, "
        f"residence cells = {np.count_nonzero(slab_mask):,}, "
        f"FOV pixels = {np.count_nonzero(fov_mask):,}, "
        f"IV pixels = {np.count_nonzero(iv_mask):,}"
    )
    if finite_snr.size:
        print(
            f"{plane_name}: SNR range = "
            f"{float(np.min(finite_snr)):.6g} to "
            f"{float(np.max(finite_snr)):.6g}"
        )
    else:
        print(f"{plane_name}: no finite SNR values")

    return {
        "plane_name": plane_name,
        "plane": plane,
        "u_centres_au": u_centres_au,
        "v_centres_au": v_centres_au,
        "residence_2d": residence_2d,
        "snr": snr_flat.reshape(shape),
        "fov_mask": fov_mask.reshape(shape),
        "iv_mask": iv_mask.reshape(shape),
        "slab_half_thickness_au": slab_half_thickness_au,
        "observer_secr_km": observer,
        "earth_secr_km": earth,
        "moon_secr_km": moon,
        "selected_boresight_synodic": selected,
        "earth_radius_km": earth_radius,
        "moon_radius_km": moon_radius,
    }


def project_point_to_plane(
    position_km: ArrayLike,
    plane_origin_km: ArrayLike,
    u_axis: ArrayLike,
    v_axis: ArrayLike,
    normal_axis: ArrayLike,
) -> tuple[float, float, float]:
    relative = np.asarray(position_km, dtype=float) - np.asarray(
        plane_origin_km,
        dtype=float,
    )
    return (
        float(relative @ np.asarray(u_axis, dtype=float) / AU_KM),
        float(relative @ np.asarray(v_axis, dtype=float) / AU_KM),
        float(relative @ np.asarray(normal_axis, dtype=float) / AU_KM),
    )


def add_boresight_plane_overlays(
    axis: plt.Axes,
    fields: dict[str, object],
) -> None:
    plane = fields["plane"]
    u = np.asarray(fields["u_centres_au"], dtype=float)
    v = np.asarray(fields["v_centres_au"], dtype=float)
    u_mesh, v_mesh = np.meshgrid(u, v, indexing="ij")

    iv_mask = np.asarray(fields["iv_mask"], dtype=bool)
    if np.any(iv_mask):
        axis.contourf(
            u_mesh,
            v_mesh,
            iv_mask.astype(float),
            levels=[0.5, 1.5],
            alpha=IV_ZONE_ALPHA,
            hatches=[IV_ZONE_HATCH],
            zorder=3,
        )

    fov_mask = np.asarray(fields["fov_mask"], dtype=bool)
    if np.any(fov_mask) and np.any(~fov_mask):
        axis.contour(
            u_mesh,
            v_mesh,
            fov_mask.astype(float),
            levels=[0.5],
            linewidths=FOV_LINEWIDTH,
            linestyles="--",
            zorder=6,
        )

    origin = np.asarray(plane["origin_km"], dtype=float)
    u_axis = np.asarray(plane["u_axis"], dtype=float)
    v_axis = np.asarray(plane["v_axis"], dtype=float)
    normal = np.asarray(plane["normal_axis"], dtype=float)

    observer_u, observer_v, observer_w = project_point_to_plane(
        fields["observer_secr_km"],
        origin,
        u_axis,
        v_axis,
        normal,
    )
    view_type = str(plane["view_type"])
    if view_type == "longitudinal":
        axis.plot(
            [max(0.0, float(u[0])), float(u[-1])],
            [0.0, 0.0],
            linewidth=BORESIGHT_LINEWIDTH,
            zorder=8,
            label="Selected boresight",
        )
    else:
        axis.scatter(
            [0.0],
            [0.0],
            marker=r"$\odot$",
            s=90,
            zorder=8,
            label="Selected boresight through plane",
        )

    # The spacecraft and body markers are orthographic projections onto the
    # plane. Labels disclose when a body centre is materially out of plane.
    axis.scatter(
        [observer_u],
        [observer_v],
        marker="*",
        s=SPACECRAFT_MARKER_SIZE,
        edgecolors="black",
        facecolors="white",
        linewidths=0.9,
        zorder=12,
    )
    if ANNOTATE_SCENE_OBJECTS:
        observer_label = "Spacecraft"
        if abs(observer_w) > 1.0e-10:
            observer_label += " (projected)"
        axis.annotate(
            observer_label,
            xy=(observer_u, observer_v),
            xytext=(6, -8),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="top",
            zorder=13,
        )

    body_specs = (
        ("Earth", fields["earth_secr_km"], fields["earth_radius_km"]),
        ("Moon", fields["moon_secr_km"], fields["moon_radius_km"]),
    )
    for name, position, radius_km in body_specs:
        body_u, body_v, body_w = project_point_to_plane(
            position,
            origin,
            u_axis,
            v_axis,
            normal,
        )
        radius_au = float(radius_km) / AU_KM
        if abs(body_w) <= radius_au:
            section_radius_au = np.sqrt(
                max(radius_au ** 2 - body_w ** 2, 0.0)
            )
            axis.add_patch(
                Circle(
                    (body_u, body_v),
                    section_radius_au,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=BODY_OUTLINE_LINEWIDTH,
                    zorder=9,
                )
            )
        axis.scatter(
            [body_u],
            [body_v],
            s=20,
            edgecolors="black",
            facecolors="white",
            linewidths=0.8,
            zorder=10,
        )
        if ANNOTATE_SCENE_OBJECTS:
            label = name if abs(body_w) <= radius_au else f"{name} (projected)"
            axis.annotate(
                label,
                xy=(body_u, body_v),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                zorder=11,
            )


def plot_boresight_aligned_plane(fields: dict[str, object]) -> None:
    """Plot residence slab with SNR, FOV, and IV-zone overlays."""

    plane = fields["plane"]
    u = np.asarray(fields["u_centres_au"], dtype=float)
    v = np.asarray(fields["v_centres_au"], dtype=float)
    u_mesh, v_mesh = np.meshgrid(u, v, indexing="ij")

    residence = np.asarray(fields["residence_2d"], dtype=float)
    positive = residence[residence > 0.0]
    if positive.size == 0:
        raise ValueError(
            f"The {fields['plane_name']} residence slab is empty."
        )

    if RESIDENCE_COLOR_SCALE == "log":
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
        levels = np.geomspace(vmin, vmax, RESIDENCE_FILLED_LEVELS)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = 0.0
        vmax = float(np.max(positive))
        levels = np.linspace(vmin, vmax, RESIDENCE_FILLED_LEVELS)
        norm = Normalize(vmin=vmin, vmax=vmax)

    figure, axis = plt.subplots(figsize=(7.2, 5.9))
    filled = axis.contourf(
        u_mesh,
        v_mesh,
        np.ma.masked_less_equal(residence, 0.0),
        levels=levels,
        norm=norm,
        extend="max",
        zorder=1,
    )
    colourbar = figure.colorbar(filled, ax=axis)
    colourbar.set_label("Residence time in finite slab (days)")

    snr = np.asarray(fields["snr"], dtype=float)
    finite_snr = snr[np.isfinite(snr)]
    if finite_snr.size:
        available_levels = [
            level
            for level in SNR_CONTOUR_LEVELS
            if float(np.min(finite_snr))
            <= float(level)
            <= float(np.max(finite_snr))
        ]
        if available_levels:
            lines = axis.contour(
                u_mesh,
                v_mesh,
                np.ma.masked_invalid(snr),
                levels=available_levels,
                linewidths=SNR_CONTOUR_LINEWIDTH,
                zorder=5,
            )
            axis.clabel(
                lines,
                inline=True,
                fontsize=8,
                fmt=lambda value: f"SNR {value:g}",
            )

    add_boresight_plane_overlays(axis, fields)

    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.set_ylim(float(v[0]), float(v[-1]))
    axis.set_xlabel(f"{plane['u_label']} (AU)")
    axis.set_ylabel(f"{plane['v_label']} (AU)")
    axis.set_aspect("equal", adjustable="box")

    title = str(plane["title"])
    if plane["cross_range_km"] is not None:
        title += (
            f"; range={float(plane['cross_range_km']) / AU_KM:.5f} AU"
        )
    title += (
        f"; residence slab +/-{float(fields['slab_half_thickness_au']):.5f} AU"
    )
    axis.set_title(title)
    figure.tight_layout()
    save_figure(figure, str(fields["plane_name"]))


# =============================================================================
# Summary and main
# =============================================================================


def save_summary(example: dict[str, object]) -> None:
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
        "residence_grid_csv": example["residence_path"],
        "residence_domain_half_width_au": SPATIAL_HALF_WIDTH_AU,
        "jd_tdb": geometry.jd_tdb,
        "observer_position_secr_km": geometry.observer_position_synodic_km,
        "iv_zone_axis_secr": geometry.invisibility_zone_axis_synodic,
        "iv_zone_half_angle_deg": np.rad2deg(
            geometry.invisibility_zone_half_angle_rad
        ),
        "pointing_config": asdict(config),
        "reference_payload": asdict(scenario.payload),
        "reference_asteroid": asdict(scenario.asteroid),
        "selected_status": result.status.value,
        "selected_boresight_secr": result.boresight_synodic,
        "selected_boresight_geo_eme": result.boresight_inertial,
        "selected_score_days": result.residence_score_days,
        "selected_geometric_cells": result.number_of_cells_in_selected_fov,
        "selected_detectable_cells": result.number_of_detectable_cells,
        "candidate_count": result.number_of_candidate_boresights,
        "feasible_candidate_count": result.number_of_feasible_boresights,
        "snr_contour_levels": SNR_CONTOUR_LEVELS,
        "visualization_plane_grid_shape": PLANE_GRID_SHAPE,
        "residence_slice_initial_half_thickness_au": (
            RESIDENCE_SLICE_INITIAL_HALF_THICKNESS_AU
        ),
        "residence_slice_max_half_thickness_au": (
            RESIDENCE_SLICE_MAX_HALF_THICKNESS_AU
        ),
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary_path = (
        OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_visualization_summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(payload), stream, indent=2)


def main() -> None:
    validate_configuration()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Loading the configured residence grid and running optimization...")
    example = build_single_epoch_example()

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

    print("Constructing boresight-aligned visualization planes...")
    plane_definitions = build_boresight_plane_definitions(
        residence_grid=residence_grid,
        geometry=geometry,
        selected_boresight_synodic=result.boresight_synodic,
        config=config,
    )

    for plane_name, plane_definition in plane_definitions.items():
        print(f"Evaluating {plane_name} plane...")
        plane_fields = evaluate_boresight_aligned_plane(
            plane_name=plane_name,
            plane=plane_definition,
            residence_grid=residence_grid,
            geometry=geometry,
            selected_boresight_synodic=result.boresight_synodic,
            snr_evaluator=snr_evaluator,
            config=config,
            scenario=scenario,
        )
        print(f"Plotting {plane_name} plane...")
        plot_boresight_aligned_plane(plane_fields)

    save_summary(example)

    print("Visualization workflow completed successfully.")
    print(f"Residence CSV: {example['residence_path']}")
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


if __name__ == "__main__":
    main()
