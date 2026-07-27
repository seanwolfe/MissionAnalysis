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

2. Three Cartesian views (x-y, x-z, and y-z), each containing:
       - the complete residence-time distribution projected through the
         omitted Cartesian axis;
       - selected-boresight SNR contour lines at SNR = 1, 2, 3, 4, and 5,
         evaluated on a plane through the spacecraft;
       - the selected boresight and selected FOV boundary;
       - the spacecraft, Earth, and Moon;
       - the Earth--Moon invisibility-zone cross-section.

The residence histogram domain is fixed to +/-0.01 AU in x, y, and z. The
input residence CSV is never created or overwritten by this script.

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
    r"tbo_residence_time_results_2/xyz_synthetic_residence_grid_sparse.csv"
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
MAXIMUM_BORESIGHT_CHANGE_DEG = 15.0
CANDIDATE_ANGULAR_SPACING_DEG = 1.5
IV_ZONE_MARGIN_DEG = 1.0

ENABLE_PREVIOUS_SCORE_SHORTCUT = False
SHORTCUT_MINIMUM_PREVIOUS_SCORE_FRACTION = 0.95

SNR_BATCH_SIZE = 20_000
CANDIDATE_BATCH_SIZE = 256
LOS_BATCH_SIZE = 20_000

# Initial direction relative to the IV-zone axis.
INITIAL_BORESIGHT_OFF_AXIS_DEG = 25.0
INITIAL_BORESIGHT_AZIMUTH_DEG = 0.0

# -----------------------------------------------------------------------------
# Angular score map
# -----------------------------------------------------------------------------
ANGLE_MAP_OFF_AXIS_MIN_DEG = 0.0
ANGLE_MAP_OFF_AXIS_MAX_DEG = 55.0
ANGLE_MAP_OFF_AXIS_SAMPLES = 25
ANGLE_MAP_AZIMUTH_MIN_DEG = -180.0
ANGLE_MAP_AZIMUTH_MAX_DEG = 180.0
ANGLE_MAP_AZIMUTH_SAMPLES = 45
ANGLE_MAP_SCORE_CONTOURS = 8

# -----------------------------------------------------------------------------
# Spatial domain and sampling
# -----------------------------------------------------------------------------
# The residence distribution is defined on +/-0.01 AU in all three axes.
SPATIAL_HALF_WIDTH_AU = 0.01
SPATIAL_HALF_WIDTH_KM = SPATIAL_HALF_WIDTH_AU * AU_KM

# Histogram samples for the full 3D residence projection. Runtime and memory
# scale with the product. SNR is evaluated only on three 2D slices.
GRID_SHAPE = (41, 41, 41)

# Small visual padding lets markers at the exact edge remain visible. It does
# not change the +/-0.01 AU residence histogram domain.
PLOT_PADDING_FRACTION = 0.01

# Smoothing affects only the rendered residence background, not optimization.
RESIDENCE_GAUSSIAN_SIGMA_VOXELS = 10.0
RESIDENCE_COLOR_SCALE = "linear"  # "linear" or "log"
RESIDENCE_FILLED_LEVELS = 50

# The SNR slice for each Cartesian view passes through the spacecraft along the
# omitted coordinate. Contours outside the selected FOV are suppressed because
# those directions are not observable by the selected pointing command.
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


def make_fixed_axes() -> tuple[FloatArray, FloatArray, FloatArray]:
    axes = tuple(
        np.linspace(-SPATIAL_HALF_WIDTH_KM, SPATIAL_HALF_WIDTH_KM, int(n))
        for n in GRID_SHAPE
    )
    return axes  # type: ignore[return-value]


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
    if len(GRID_SHAPE) != 3 or any(int(value) < 3 for value in GRID_SHAPE):
        raise ValueError("GRID_SHAPE must contain three integers >= 3.")
    if SPATIAL_HALF_WIDTH_AU <= 0.0:
        raise ValueError("SPATIAL_HALF_WIDTH_AU must be positive.")
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
        OBSERVER_DISTANCE_TOWARD_SUN_KM * sun_direction_geo_eme
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
# Residence volume and SNR slices
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


def build_residence_volume(
    residence_grid: optimizer.ResidenceGrid,
) -> dict[str, object]:
    """Deposit the complete sparse grid into the fixed +/-0.01 AU volume."""

    x_km, y_km, z_km = make_fixed_axes()
    residence_positions = np.asarray(
        residence_grid.positions_synodic_km,
        dtype=float,
    )
    inside_domain = np.all(
        np.abs(residence_positions) <= SPATIAL_HALF_WIDTH_KM + 1.0e-9,
        axis=1,
    )
    number_outside = int(np.count_nonzero(~inside_domain))
    if number_outside:
        print(
            f"Warning: {number_outside:,} residence cells lie outside the "
            "+/-0.01 AU plotting volume and will not appear in the spatial "
            "figures. They remain included in the optimizer."
        )

    residence_3d, _ = np.histogramdd(
        residence_positions,
        bins=(
            centre_axis_to_edges(x_km),
            centre_axis_to_edges(y_km),
            centre_axis_to_edges(z_km),
        ),
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
        "residence_3d": residence_3d,
    }


def make_slice_positions(
    horizontal_values_km: FloatArray,
    vertical_values_km: FloatArray,
    horizontal_axis: int,
    vertical_axis: int,
    omit_axis: int,
    fixed_coordinate_km: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    horizontal_mesh, vertical_mesh = np.meshgrid(
        horizontal_values_km,
        vertical_values_km,
        indexing="ij",
    )
    positions = np.empty((horizontal_mesh.size, 3), dtype=float)
    positions[:, horizontal_axis] = horizontal_mesh.ravel()
    positions[:, vertical_axis] = vertical_mesh.ravel()
    positions[:, omit_axis] = float(fixed_coordinate_km)
    return positions, horizontal_mesh, vertical_mesh


def evaluate_selected_boresight_slice(
    projection_name: str,
    volume: dict[str, object],
    geometry: optimizer.SingleEpochSearchGeometry,
    selected_boresight_synodic: FloatArray,
    snr_evaluator: optimizer.SNREvaluator,
    config: optimizer.PointingConfig,
    scenario: reference_payload.ReferencePayloadScenario,
) -> dict[str, object]:
    """Evaluate fixed-boresight SNR on a Cartesian plane through spacecraft."""

    specification = PROJECTIONS[projection_name]
    horizontal_axis = int(specification["horizontal_axis"])
    vertical_axis = int(specification["vertical_axis"])
    omit_axis = int(specification["omit_axis"])

    axes = [
        np.asarray(volume["x_km"], dtype=float),
        np.asarray(volume["y_km"], dtype=float),
        np.asarray(volume["z_km"], dtype=float),
    ]
    horizontal = axes[horizontal_axis]
    vertical = axes[vertical_axis]

    observer = np.asarray(
        geometry.observer_position_synodic_km,
        dtype=float,
    )
    fixed_coordinate = float(observer[omit_axis])
    positions, horizontal_mesh, vertical_mesh = make_slice_positions(
        horizontal,
        vertical,
        horizontal_axis,
        vertical_axis,
        omit_axis,
        fixed_coordinate,
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

    evaluate_mask = valid & outside_bodies
    snr_flat = np.full(positions.shape[0], np.nan, dtype=float)
    indices = np.flatnonzero(evaluate_mask)
    if indices.size:
        snr_flat[indices] = optimizer.evaluate_snr_in_batches(
            positions_synodic_km=positions[indices],
            boresights_synodic=selected,
            snr_evaluator=snr_evaluator,
            batch_size=config.snr_batch_size,
        )

    if MASK_SNR_CONTOURS_OUTSIDE_FOV:
        snr_flat[~fov_mask] = np.nan

    shape = horizontal_mesh.shape
    return {
        "horizontal_km": horizontal,
        "vertical_km": vertical,
        "snr": snr_flat.reshape(shape),
        "fov_mask": fov_mask.reshape(shape),
        "iv_mask": iv_mask.reshape(shape),
        "slice_coordinate_km": fixed_coordinate,
        "observer_secr_km": observer,
        "earth_secr_km": earth,
        "moon_secr_km": moon,
        "selected_boresight_synodic": selected,
        "earth_radius_km": earth_radius,
        "moon_radius_km": moon_radius,
    }


def project_residence(
    residence_3d: FloatArray,
    omit_axis: int,
) -> FloatArray:
    return np.asarray(np.nansum(residence_3d, axis=omit_axis), dtype=float)


def line_to_axes_boundary(
    origin_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    x0, y0 = origin_xy
    dx, dy = direction_xy
    norm = float(np.hypot(dx, dy))
    if norm <= 1.0e-14:
        return None
    dx /= norm
    dy /= norm

    distances: list[float] = []
    if dx > 0.0:
        distances.append((x_limits[1] - x0) / dx)
    elif dx < 0.0:
        distances.append((x_limits[0] - x0) / dx)
    if dy > 0.0:
        distances.append((y_limits[1] - y0) / dy)
    elif dy < 0.0:
        distances.append((y_limits[0] - y0) / dy)

    positive = [value for value in distances if value > 0.0]
    if not positive:
        return None
    distance = min(positive)
    return (x0, y0), (x0 + distance * dx, y0 + distance * dy)


def add_scene_overlays(
    axis: plt.Axes,
    horizontal_scaled: FloatArray,
    vertical_scaled: FloatArray,
    slice_fields: dict[str, object],
    horizontal_axis: int,
    vertical_axis: int,
) -> None:
    horizontal_mesh, vertical_mesh = np.meshgrid(
        horizontal_scaled,
        vertical_scaled,
        indexing="ij",
    )

    iv_mask = np.asarray(slice_fields["iv_mask"], dtype=bool)
    if np.any(iv_mask):
        axis.contourf(
            horizontal_mesh,
            vertical_mesh,
            iv_mask.astype(float),
            levels=[0.5, 1.5],
            alpha=IV_ZONE_ALPHA,
            hatches=[IV_ZONE_HATCH],
            zorder=3,
        )

    fov_mask = np.asarray(slice_fields["fov_mask"], dtype=bool)
    if np.any(fov_mask) and np.any(~fov_mask):
        axis.contour(
            horizontal_mesh,
            vertical_mesh,
            fov_mask.astype(float),
            levels=[0.5],
            linewidths=FOV_LINEWIDTH,
            linestyles="--",
            zorder=6,
        )

    observer = np.asarray(slice_fields["observer_secr_km"], dtype=float)
    earth = np.asarray(slice_fields["earth_secr_km"], dtype=float)
    moon = np.asarray(slice_fields["moon_secr_km"], dtype=float)
    boresight = np.asarray(
        slice_fields["selected_boresight_synodic"],
        dtype=float,
    )

    observer_xy = (
        float(observer[horizontal_axis] / PLOT_DISTANCE_SCALE_KM),
        float(observer[vertical_axis] / PLOT_DISTANCE_SCALE_KM),
    )
    plot_half_width = (
        SPATIAL_HALF_WIDTH_AU * (1.0 + PLOT_PADDING_FRACTION)
    )
    ray = line_to_axes_boundary(
        observer_xy,
        (
            float(boresight[horizontal_axis]),
            float(boresight[vertical_axis]),
        ),
        (-plot_half_width, plot_half_width),
        (-plot_half_width, plot_half_width),
    )
    if ray is not None:
        (x0, y0), (x1, y1) = ray
        axis.plot(
            [x0, x1],
            [y0, y1],
            linewidth=BORESIGHT_LINEWIDTH,
            zorder=8,
            label="Selected boresight",
        )

    body_specs = (
        ("Earth", earth, float(slice_fields["earth_radius_km"])),
        ("Moon", moon, float(slice_fields["moon_radius_km"])),
    )
    for name, position, radius_km in body_specs:
        x_body = float(position[horizontal_axis] / PLOT_DISTANCE_SCALE_KM)
        y_body = float(position[vertical_axis] / PLOT_DISTANCE_SCALE_KM)
        axis.add_patch(
            Circle(
                (x_body, y_body),
                radius_km / PLOT_DISTANCE_SCALE_KM,
                facecolor="none",
                edgecolor="black",
                linewidth=BODY_OUTLINE_LINEWIDTH,
                zorder=9,
            )
        )
        axis.scatter(
            [x_body],
            [y_body],
            s=20,
            edgecolors="black",
            facecolors="white",
            linewidths=0.8,
            zorder=10,
        )
        if ANNOTATE_SCENE_OBJECTS:
            axis.annotate(
                name,
                xy=(x_body, y_body),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                zorder=11,
            )

    axis.scatter(
        [observer_xy[0]],
        [observer_xy[1]],
        marker="*",
        s=SPACECRAFT_MARKER_SIZE,
        edgecolors="black",
        facecolors="white",
        linewidths=0.9,
        zorder=12,
        clip_on=False,
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
            zorder=13,
        )


def plot_combined_projection(
    projection_name: str,
    volume: dict[str, object],
    slice_fields: dict[str, object],
) -> None:
    """Plot residence projection with selected-boresight SNR contours."""

    specification = PROJECTIONS[projection_name]
    horizontal_axis = int(specification["horizontal_axis"])
    vertical_axis = int(specification["vertical_axis"])
    omit_axis = int(specification["omit_axis"])

    axes = [
        np.asarray(volume["x_km"], dtype=float),
        np.asarray(volume["y_km"], dtype=float),
        np.asarray(volume["z_km"], dtype=float),
    ]
    horizontal = axes[horizontal_axis] / PLOT_DISTANCE_SCALE_KM
    vertical = axes[vertical_axis] / PLOT_DISTANCE_SCALE_KM
    horizontal_mesh, vertical_mesh = np.meshgrid(
        horizontal,
        vertical,
        indexing="ij",
    )

    residence_projection = project_residence(
        np.asarray(volume["residence_3d"], dtype=float),
        omit_axis,
    )
    positive = residence_projection[residence_projection > 0.0]
    if positive.size == 0:
        raise ValueError(
            f"The {projection_name} projected residence field is empty."
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

    residence_for_plot = np.ma.masked_less_equal(
        residence_projection,
        0.0,
    )

    figure, axis = plt.subplots(figsize=(7.0, 5.8))
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

    snr = np.asarray(slice_fields["snr"], dtype=float)
    finite_snr = snr[np.isfinite(snr)]
    if finite_snr.size:
        available_levels = [
            level
            for level in SNR_CONTOUR_LEVELS
            if float(np.min(finite_snr)) <= level <= float(np.max(finite_snr))
        ]
        if available_levels:
            lines = axis.contour(
                horizontal_mesh,
                vertical_mesh,
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

    add_scene_overlays(
        axis=axis,
        horizontal_scaled=horizontal,
        vertical_scaled=vertical,
        slice_fields=slice_fields,
        horizontal_axis=horizontal_axis,
        vertical_axis=vertical_axis,
    )

    padded_half_width = (
        SPATIAL_HALF_WIDTH_AU * (1.0 + PLOT_PADDING_FRACTION)
    )
    axis.set_xlim(-padded_half_width, padded_half_width)
    axis.set_ylim(-padded_half_width, padded_half_width)
    axis.set_xlabel(
        f"{specification['horizontal_label']} ({PLOT_DISTANCE_UNIT_LABEL})"
    )
    axis.set_ylabel(
        f"{specification['vertical_label']} ({PLOT_DISTANCE_UNIT_LABEL})"
    )
    axis.set_aspect("equal", adjustable="box")
    fixed_coordinate_au = (
        float(slice_fields["slice_coordinate_km"]) / AU_KM
    )
    omitted_label = ("x", "y", "z")[omit_axis]
    axis.set_title(
        f"Residence projection with selected-boresight SNR contours "
        f"({projection_name}; {omitted_label}={fixed_coordinate_au:.4f} AU slice)"
    )
    figure.tight_layout()
    save_figure(figure, f"residence_snr_{projection_name}")


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

    print("Building the complete +/-0.01 AU residence volume...")
    volume = build_residence_volume(residence_grid)

    for projection_name in PROJECTIONS:
        print(f"Evaluating selected-boresight SNR slice for {projection_name}...")
        slice_fields = evaluate_selected_boresight_slice(
            projection_name=projection_name,
            volume=volume,
            geometry=geometry,
            selected_boresight_synodic=result.boresight_synodic,
            snr_evaluator=snr_evaluator,
            config=config,
            scenario=scenario,
        )
        print(f"Plotting combined residence/SNR view for {projection_name}...")
        plot_combined_projection(
            projection_name=projection_name,
            volume=volume,
            slice_fields=slice_fields,
        )

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
