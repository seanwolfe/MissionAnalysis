from __future__ import annotations

"""
Generate three-dimensional asteroid SNR maps and contour slices.

This script evaluates ``compute_asteroid_snr`` on a Cartesian x-y-z grid for
one representative payload, asteroid, observer, and environmental geometry.
It imports the SNR implementation from:

    payload_asteroid_snr_model.py

Place that file in the same directory as this script, or otherwise make it
available on Python's import path.

Interpretation of the grid
--------------------------
Each grid point is an independent hypothetical asteroid observation. The
observer, Sun, Earth, and Moon states are held fixed at the representative
epoch. The asteroid position is replaced by the grid-point position, while its
velocity and intrinsic properties are held at the representative values below.

At every grid point, the SNR model derives the instantaneous apparent angular
speed from the relative observer/asteroid states and assumes that speed is
constant over one exposure of ``payload.exposure_time_s``. Adjacent grid points
do not represent consecutive samples within an exposure.

Boresight modes
---------------
``BORESIGHT_MODE = "point_to_grid"``
    The payload is pointed directly at each candidate asteroid position. This
    is generally the most useful mode for a three-dimensional *detectability*
    map because the candidate asteroid is on-axis at every grid point and the
    Earth/Moon stray-light angles vary with target direction.

``BORESIGHT_MODE = "fixed"``
    One fixed inertial boresight is used throughout the grid. The underlying
    SNR model does not apply an FOV test or a field-dependent direct-source
    vignetting model, so this mode is an SNR-only diagnostic and must not be
    interpreted as a complete detection map outside the payload FOV.

Outputs
-------
1. A compressed NPZ file containing the x, y, and z axes and the complete
   three-dimensional SNR grid. Optional diagnostic grids are also saved.
2. Three separate filled-contour figures:
       - x-y plane at the requested z slice
       - x-z plane at the requested y slice
       - y-z plane at the requested x slice
3. Optional three-dimensional SNR isosurfaces when scikit-image is installed
   and ``MAKE_3D_ISOSURFACES`` is enabled.

Dependencies
------------
Required:
    numpy
    scipy
    matplotlib

Optional for true three-dimensional isosurfaces:
    scikit-image

Run directly in PyCharm using the green Run button. Edit only the
CONFIGURATION section unless you want to change the workflow itself.
"""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from numpy.typing import NDArray

from payload_asteroid_snr_model import (
    AsteroidProperties,
    EnvironmentConfig,
    HG12PhaseModel,
    ObservationGeometry,
    PayloadConfig,
    SNROptions,
    compute_asteroid_snr,
)


FloatArray = NDArray[np.float64]


# =============================================================================
# CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
OUTPUT_DIRECTORY = Path("snr_contour_output")
OUTPUT_STEM = "average_scenario_snr"
SAVE_FULL_GRID_NPZ = True
SAVE_DIAGNOSTIC_GRIDS = True
SAVE_PNG = True
SAVE_PDF = True
FIGURE_DPI = 300
SHOW_FIGURES = True

# -----------------------------------------------------------------------------
# Representative payload
# -----------------------------------------------------------------------------

VEGA_FLUX_DENSITY = 3.68e-2
REFERENCE_WAVELENGTH_M = 0.555e-6

# Replace these preliminary values with the average/reference payload scenario.
PAYLOAD = PayloadConfig(
    exposure_time_s=30.0,
    aperture_diameter_m=0.30,
    focal_length_m=0.78,
    pixel_scale_arcsec_per_px=1.0,
    pixel_pitch_m=3.76e-6,
    psf_sigma_px=1.2,

    # Scalars are treated as constant across the wavelength band. They may be
    # replaced by one-dimensional arrays sampled on wavelength_m.
    quantum_efficiency=0.80,
    optical_throughput=0.80,

    dark_current_e_per_s_px=0.002,
    read_noise_e_rms_per_px=1.2,
    background_surface_brightness_mag_arcsec2=22.0,

    wavelength_lower_m=400.0e-9,
    wavelength_upper_m=800.0e-9,
    spectral_samples=2001,

    # zero_point_mag is the magnitude that produces 1 electron/s. When it is
    # supplied, Vega fields are not required.
    zero_point_mag=None,

    # Vega spectral energy flux density at the reference wavelength.
    # Units: W m^-2 m^-1
    vega_flux_density_w_m2_m=VEGA_FLUX_DENSITY,

    # Wavelength at which the scalar Vega flux density is defined.
    vega_reference_wavelength_m=REFERENCE_WAVELENGTH_M,
)

# -----------------------------------------------------------------------------
# Representative asteroid
# -----------------------------------------------------------------------------
# These properties are held constant throughout the complete spatial grid.
ASTEROID = AsteroidProperties(
    absolute_magnitude=30.0,
    geometric_albedo=0.15,
    g12=0.40,
)

# The average asteroid inertial velocity is also held constant throughout the
# grid. The apparent angular speed still changes with candidate position.
ASTEROID_VELOCITY_KM_S = np.array([0.15, 0.45, 0.02], dtype=float)

# Optional angular-rate override. Leave as None to derive the rate from the
# observer and asteroid relative states at each grid point. A scalar override
# applies the same angular speed everywhere.
ASTEROID_ANGULAR_RATE_OVERRIDE_ARCSEC_S: float | None = None

# -----------------------------------------------------------------------------
# Representative observer and environment state
# -----------------------------------------------------------------------------
# All positions and velocities must use the same origin, axes, frame, and
# representative epoch. Positions are km and velocities are km/s.
OBSERVER_POSITION_KM = np.array([-1.50e6, 8.0e5, 0.0e5], dtype=float)
OBSERVER_VELOCITY_KM_S = np.array([0.0, -0.20, 0.0], dtype=float)

SUN_POSITION_KM = np.array([-149_597_870.7, 0.0, 0.0], dtype=float)
EARTH_POSITION_KM = np.array([0.0, 0.0, 0.0], dtype=float)
MOON_POSITION_KM = np.array([0.0, 384_400.0, 0.0], dtype=float)

ENVIRONMENT = EnvironmentConfig()
SNR_OPTIONS = SNROptions(
    include_earth_double_reflection=True,
    include_moon_double_reflection=True,
    include_earth_stray_light=True,
    include_moon_stray_light=True,
    aperture_pixel_mode="continuous",
    validate_inputs=True,
)

# -----------------------------------------------------------------------------
# Payload boresight
# -----------------------------------------------------------------------------
BORESIGHT_MODE: Literal["point_to_grid", "fixed"] = "point_to_grid"

# Used only when BORESIGHT_MODE == "fixed". It need not be normalized here.
FIXED_BORESIGHT_VECTOR = np.array([1.0, 0.0, 0.0], dtype=float)

# -----------------------------------------------------------------------------
# Three-dimensional spatial grid
# -----------------------------------------------------------------------------
# The generated Cartesian axes are absolute positions:
#
#     absolute_position = GRID_CENTER_KM + local_grid_offset
#
# The plots show local offsets from GRID_CENTER_KM, which usually makes the
# axes easier to interpret.
GRID_CENTER_KM = np.array([0, 0.0, 0.0], dtype=float)
GRID_HALF_WIDTH_KM = np.array([1_500_000.0, 1_000_000.0, 1_000_000.0], dtype=float)

# Number of samples along x, y, and z. Runtime and storage scale with the
# product nx * ny * nz. A 61^3 grid has 226,981 points.
GRID_SHAPE = (41, 41, 31)

# Grid points are processed in chunks to avoid constructing all batch inputs at
# once. The full SNR output grid still resides in memory.
CHUNK_SIZE = 20_000

# Mask physically invalid grid points before calling the SNR model.
MIN_ASTEROID_OBSERVER_DISTANCE_KM = 1.0
MASK_INSIDE_EARTH = True
MASK_INSIDE_MOON = True
MASK_INSIDE_SUN = True

# -----------------------------------------------------------------------------
# Orthogonal contour slices
# -----------------------------------------------------------------------------
# Slice coordinates are local offsets from GRID_CENTER_KM. The nearest grid
# plane is used and the actual selected coordinate is reported.
SLICE_X_OFFSET_KM = 0.0  # y-z contour
SLICE_Y_OFFSET_KM = 0.0  # x-z contour
SLICE_Z_OFFSET_KM = 0.0e5  # x-y contour

# Plot distance scaling. For example, 1000.0 displays axes in 10^3 km.
PLOT_DISTANCE_SCALE_KM = 100000.0
PLOT_DISTANCE_UNIT_LABEL = r"$10^5$ km"

# SNR colour scale. Log scaling is generally more informative when the SNR
# spans several orders of magnitude.
COLOR_SCALE: Literal["log", "linear"] = "linear"
N_FILLED_CONTOUR_LEVELS = 40
PLOT_SNR_MIN: float | None = None
PLOT_SNR_MAX: float | None = None

# Optional labelled SNR contour lines overlaid on each filled contour. Levels
# outside a slice's finite data range are skipped automatically.
SNR_LINE_LEVELS = (1.0, 3.0, 5.0, 10.0)

# -----------------------------------------------------------------------------
# Optional true three-dimensional contour surfaces
# -----------------------------------------------------------------------------
# Requires: pip install scikit-image
MAKE_3D_ISOSURFACES = False
SNR_ISOSURFACE_LEVELS = (1.0, 3.0, 5.0, 10.0)

# =============================================================================


def validate_configuration() -> None:
    """Validate the contour-script configuration before expensive work."""

    if BORESIGHT_MODE not in {"point_to_grid", "fixed"}:
        raise ValueError(
            "BORESIGHT_MODE must be 'point_to_grid' or 'fixed'."
        )

    if COLOR_SCALE not in {"log", "linear"}:
        raise ValueError("COLOR_SCALE must be 'log' or 'linear'.")

    if len(GRID_SHAPE) != 3 or any(int(value) < 2 for value in GRID_SHAPE):
        raise ValueError("GRID_SHAPE must contain three integers >= 2.")

    if np.asarray(GRID_CENTER_KM).shape != (3,):
        raise ValueError("GRID_CENTER_KM must have shape (3,).")

    if np.asarray(GRID_HALF_WIDTH_KM).shape != (3,):
        raise ValueError("GRID_HALF_WIDTH_KM must have shape (3,).")

    if np.any(np.asarray(GRID_HALF_WIDTH_KM, dtype=float) <= 0.0):
        raise ValueError("Every GRID_HALF_WIDTH_KM component must be positive.")

    if CHUNK_SIZE < 1:
        raise ValueError("CHUNK_SIZE must be >= 1.")

    if PLOT_DISTANCE_SCALE_KM <= 0.0:
        raise ValueError("PLOT_DISTANCE_SCALE_KM must be positive.")

    if MIN_ASTEROID_OBSERVER_DISTANCE_KM <= 0.0:
        raise ValueError(
            "MIN_ASTEROID_OBSERVER_DISTANCE_KM must be positive."
        )

    if PLOT_SNR_MIN is not None and PLOT_SNR_MIN <= 0.0:
        raise ValueError("PLOT_SNR_MIN must be positive when specified.")

    if (
        PLOT_SNR_MIN is not None
        and PLOT_SNR_MAX is not None
        and PLOT_SNR_MAX <= PLOT_SNR_MIN
    ):
        raise ValueError("PLOT_SNR_MAX must exceed PLOT_SNR_MIN.")


def make_grid_axes() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Create absolute x, y, and z axes in kilometres."""

    centre = np.asarray(GRID_CENTER_KM, dtype=float)
    half_width = np.asarray(GRID_HALF_WIDTH_KM, dtype=float)
    nx, ny, nz = (int(value) for value in GRID_SHAPE)

    x_km = np.linspace(
        centre[0] - half_width[0],
        centre[0] + half_width[0],
        nx,
    )
    y_km = np.linspace(
        centre[1] - half_width[1],
        centre[1] + half_width[1],
        ny,
    )
    z_km = np.linspace(
        centre[2] - half_width[2],
        centre[2] + half_width[2],
        nz,
    )

    return x_km, y_km, z_km


def flat_grid_positions(
    x_km: FloatArray,
    y_km: FloatArray,
    z_km: FloatArray,
) -> FloatArray:
    """Return all grid positions as an array with shape (n_points, 3)."""

    x_grid, y_grid, z_grid = np.meshgrid(
        x_km,
        y_km,
        z_km,
        indexing="ij",
    )
    return np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
            z_grid.ravel(),
        )
    )


def make_validity_mask(asteroid_positions_km: FloatArray) -> NDArray[np.bool_]:
    """Mask grid points where the photometric geometry is not meaningful."""

    positions = np.asarray(asteroid_positions_km, dtype=float)
    valid = np.all(np.isfinite(positions), axis=-1)

    observer_range_km = np.linalg.norm(
        positions - np.asarray(OBSERVER_POSITION_KM, dtype=float),
        axis=-1,
    )
    valid &= observer_range_km > MIN_ASTEROID_OBSERVER_DISTANCE_KM

    if MASK_INSIDE_EARTH:
        earth_range_km = np.linalg.norm(
            positions - np.asarray(EARTH_POSITION_KM, dtype=float),
            axis=-1,
        )
        valid &= earth_range_km > ENVIRONMENT.earth.radius_km

    if MASK_INSIDE_MOON:
        moon_range_km = np.linalg.norm(
            positions - np.asarray(MOON_POSITION_KM, dtype=float),
            axis=-1,
        )
        valid &= moon_range_km > ENVIRONMENT.moon.radius_km

    if MASK_INSIDE_SUN:
        sun_range_km = np.linalg.norm(
            positions - np.asarray(SUN_POSITION_KM, dtype=float),
            axis=-1,
        )
        valid &= sun_range_km > ENVIRONMENT.solar_radius_km

    return valid


def make_boresights(asteroid_positions_km: FloatArray) -> FloatArray:
    """Construct one boresight vector for every candidate asteroid point."""

    n_points = asteroid_positions_km.shape[0]

    if BORESIGHT_MODE == "fixed":
        fixed = np.asarray(FIXED_BORESIGHT_VECTOR, dtype=float)
        norm = np.linalg.norm(fixed)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("FIXED_BORESIGHT_VECTOR must be finite and nonzero.")
        return np.broadcast_to(fixed / norm, (n_points, 3)).copy()

    line_of_sight = (
        asteroid_positions_km
        - np.asarray(OBSERVER_POSITION_KM, dtype=float)
    )
    norms = np.linalg.norm(line_of_sight, axis=-1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(
            "A point-to-grid boresight cannot be constructed at the observer."
        )
    return line_of_sight / norms


def evaluate_snr_grid(
    asteroid_positions_km: FloatArray,
) -> dict[str, FloatArray]:
    """Evaluate the SNR model on all valid grid points in chunks."""

    n_points = asteroid_positions_km.shape[0]
    valid_mask = make_validity_mask(asteroid_positions_km)

    output_names = ["snr"]
    if SAVE_DIAGNOSTIC_GRIDS:
        output_names.extend(
            [
                "apparent_magnitude",
                "apparent_angular_speed_arcsec_s",
                "trail_length_px",
                "signal_electrons",
                "total_noise_rms_e",
                "asteroid_off_axis_angle_rad",
                "earth_off_axis_angle_rad",
                "moon_off_axis_angle_rad",
            ]
        )

    outputs = {
        name: np.full(n_points, np.nan, dtype=float)
        for name in output_names
    }

    valid_indices = np.flatnonzero(valid_mask)
    phase_model = HG12PhaseModel.from_default_table()

    for start in range(0, valid_indices.size, CHUNK_SIZE):
        selected_indices = valid_indices[start : start + CHUNK_SIZE]
        positions = asteroid_positions_km[selected_indices]
        n_chunk = positions.shape[0]

        boresights = make_boresights(positions)

        angular_rate_override: float | FloatArray | None
        if ASTEROID_ANGULAR_RATE_OVERRIDE_ARCSEC_S is None:
            angular_rate_override = None
        else:
            angular_rate_override = float(
                ASTEROID_ANGULAR_RATE_OVERRIDE_ARCSEC_S
            )

        geometry = ObservationGeometry(
            observer_position_km=np.asarray(OBSERVER_POSITION_KM, dtype=float),
            observer_velocity_km_s=np.asarray(OBSERVER_VELOCITY_KM_S, dtype=float),
            asteroid_position_km=positions,
            asteroid_velocity_km_s=np.broadcast_to(
                np.asarray(ASTEROID_VELOCITY_KM_S, dtype=float),
                (n_chunk, 3),
            ),
            sun_position_km=np.asarray(SUN_POSITION_KM, dtype=float),
            earth_position_km=np.asarray(EARTH_POSITION_KM, dtype=float),
            moon_position_km=np.asarray(MOON_POSITION_KM, dtype=float),
            boresight_unit_vector=boresights,
            asteroid_angular_rate_arcsec_s=angular_rate_override,
        )

        result = compute_asteroid_snr(
            payload=PAYLOAD,
            asteroid=ASTEROID,
            geometry=geometry,
            environment=ENVIRONMENT,
            options=SNR_OPTIONS,
            phase_model=phase_model,
        )

        outputs["snr"][selected_indices] = np.asarray(
            result.snr,
            dtype=float,
        ).reshape(-1)

        if SAVE_DIAGNOSTIC_GRIDS:
            for name in output_names[1:]:
                outputs[name][selected_indices] = np.asarray(
                    getattr(result, name),
                    dtype=float,
                ).reshape(-1)

        completed = min(start + n_chunk, valid_indices.size)
        print(
            f"Evaluated {completed:,} / {valid_indices.size:,} valid "
            f"grid points"
        )

    outputs["valid_mask"] = valid_mask.astype(float)
    return outputs


def reshape_outputs(
    flat_outputs: dict[str, FloatArray],
) -> dict[str, FloatArray]:
    """Reshape every flat output to GRID_SHAPE."""

    shape = tuple(int(value) for value in GRID_SHAPE)
    return {
        name: np.asarray(values).reshape(shape)
        for name, values in flat_outputs.items()
    }


def nearest_index(axis_values: FloatArray, requested_value: float) -> int:
    """Return the index of the nearest sampled axis value."""

    return int(np.argmin(np.abs(axis_values - requested_value)))


def choose_colour_limits(values: FloatArray) -> tuple[float, float]:
    """Determine finite colour limits for one contour slice."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]

    if COLOR_SCALE == "log":
        finite = finite[finite > 0.0]

    if finite.size == 0:
        raise ValueError("The selected contour slice contains no plottable SNR values.")

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))

    vmin = data_min if PLOT_SNR_MIN is None else float(PLOT_SNR_MIN)
    vmax = data_max if PLOT_SNR_MAX is None else float(PLOT_SNR_MAX)

    if COLOR_SCALE == "log":
        vmin = max(vmin, np.finfo(float).tiny)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        # A constant slice cannot produce a valid contour interval. Expand the
        # range slightly while preserving the selected scale.
        if COLOR_SCALE == "log":
            vmin = max(vmin / 1.01, np.finfo(float).tiny)
            vmax = max(vmax * 1.01, vmin * 1.01)
        else:
            delta = max(abs(vmin) * 0.01, 1.0e-12)
            vmin -= delta
            vmax += delta

    return vmin, vmax


def make_filled_levels(vmin: float, vmax: float) -> FloatArray:
    """Create contour levels consistent with the requested colour scale."""

    if COLOR_SCALE == "log":
        return np.geomspace(vmin, vmax, N_FILLED_CONTOUR_LEVELS)
    return np.linspace(vmin, vmax, N_FILLED_CONTOUR_LEVELS)


def plot_contour_slice(
    horizontal_values: FloatArray,
    vertical_values: FloatArray,
    snr_slice: FloatArray,
    horizontal_label: str,
    vertical_label: str,
    title: str,
    output_name: str,
) -> None:
    """Create and save one filled SNR contour figure."""

    horizontal_mesh, vertical_mesh = np.meshgrid(
        horizontal_values,
        vertical_values,
        indexing="ij",
    )

    values = np.asarray(snr_slice, dtype=float)
    vmin, vmax = choose_colour_limits((0.0,5))
    levels = make_filled_levels(vmin, vmax)

    if COLOR_SCALE == "log":
        plotted_values = np.ma.masked_where(
            ~np.isfinite(values) | (values <= 0.0),
            values,
        )
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        plotted_values = np.ma.masked_invalid(values)
        norm = Normalize(vmin=vmin, vmax=vmax)

    figure, axis = plt.subplots(figsize=(7.0, 5.5))
    filled = axis.contourf(
        horizontal_mesh,
        vertical_mesh,
        plotted_values,
        levels=levels,
        norm=norm,
        extend="both",
    )

    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        line_levels = [
            float(level)
            for level in SNR_LINE_LEVELS
            if min_value <= float(level) <= max_value
        ]
        if line_levels:
            lines = axis.contour(
                horizontal_mesh,
                vertical_mesh,
                plotted_values,
                levels=line_levels,
                linewidths=0.8,
                linestyles="--",
            )
            axis.clabel(lines, inline=True, fontsize=8, fmt="%g")

    colourbar = figure.colorbar(filled, ax=axis)
    colourbar.set_label("SNR")

    axis.set_xlabel(horizontal_label)
    axis.set_ylabel(vertical_label)
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()

    if SAVE_PNG:
        figure.savefig(
            OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_{output_name}.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    if SAVE_PDF:
        figure.savefig(
            OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_{output_name}.pdf",
            bbox_inches="tight",
        )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(figure)


def save_grid_data(
    x_km: FloatArray,
    y_km: FloatArray,
    z_km: FloatArray,
    grids: dict[str, FloatArray],
) -> None:
    """Save the complete spatial grid and model results."""

    if not SAVE_FULL_GRID_NPZ:
        return

    payload = {
        "x_km": x_km,
        "y_km": y_km,
        "z_km": z_km,
        "grid_center_km": np.asarray(GRID_CENTER_KM, dtype=float),
        "grid_half_width_km": np.asarray(GRID_HALF_WIDTH_KM, dtype=float),
    }
    payload.update(grids)

    np.savez_compressed(
        OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_grid.npz",
        **payload,
    )


def json_safe(value):
    """Convert configuration values to JSON-compatible objects."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def save_run_summary(
    x_km: FloatArray,
    y_km: FloatArray,
    z_km: FloatArray,
    snr_grid: FloatArray,
    selected_indices: tuple[int, int, int],
) -> None:
    """Save the adopted scenario and summary statistics as JSON."""

    finite_snr = snr_grid[np.isfinite(snr_grid)]
    summary = {
        "payload": asdict(PAYLOAD),
        "asteroid": asdict(ASTEROID),
        "environment": asdict(ENVIRONMENT),
        "snr_options": asdict(SNR_OPTIONS),
        "observer_position_km": OBSERVER_POSITION_KM,
        "observer_velocity_km_s": OBSERVER_VELOCITY_KM_S,
        "asteroid_velocity_km_s": ASTEROID_VELOCITY_KM_S,
        "sun_position_km": SUN_POSITION_KM,
        "earth_position_km": EARTH_POSITION_KM,
        "moon_position_km": MOON_POSITION_KM,
        "boresight_mode": BORESIGHT_MODE,
        "fixed_boresight_vector": FIXED_BORESIGHT_VECTOR,
        "grid_shape": GRID_SHAPE,
        "grid_center_km": GRID_CENTER_KM,
        "grid_half_width_km": GRID_HALF_WIDTH_KM,
        "selected_slice_indices_xyz": selected_indices,
        "selected_slice_coordinates_km": {
            "x": float(x_km[selected_indices[0]]),
            "y": float(y_km[selected_indices[1]]),
            "z": float(z_km[selected_indices[2]]),
        },
        "finite_grid_points": int(finite_snr.size),
        "snr_min": float(np.min(finite_snr)) if finite_snr.size else None,
        "snr_median": float(np.median(finite_snr)) if finite_snr.size else None,
        "snr_max": float(np.max(finite_snr)) if finite_snr.size else None,
    }

    for level in SNR_LINE_LEVELS:
        key = f"fraction_snr_ge_{level:g}"
        summary[key] = (
            float(np.mean(finite_snr >= level))
            if finite_snr.size
            else None
        )

    with (
        OUTPUT_DIRECTORY / f"{OUTPUT_STEM}_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(json_safe(summary), file, indent=2)


def make_isosurface_plots(
    x_local_scaled: FloatArray,
    y_local_scaled: FloatArray,
    z_local_scaled: FloatArray,
    snr_grid: FloatArray,
) -> None:
    """Optionally create one true 3D isosurface figure per SNR level."""

    if not MAKE_3D_ISOSURFACES:
        return

    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise ImportError(
            "MAKE_3D_ISOSURFACES=True requires scikit-image. "
            "Install it with: pip install scikit-image"
        ) from exc

    finite_values = np.asarray(snr_grid, dtype=float)
    finite_mask = np.isfinite(finite_values)
    if not np.any(finite_mask):
        raise ValueError("The SNR grid contains no finite values.")

    # marching_cubes cannot accept NaN. Fill invalid regions below all positive
    # isosurface levels so they are treated as outside the detectable volume.
    finite_min = float(np.min(finite_values[finite_mask]))
    fill_value = min(0.0, finite_min)
    volume = np.where(finite_mask, finite_values, fill_value)

    spacing = (
        float(x_local_scaled[1] - x_local_scaled[0]),
        float(y_local_scaled[1] - y_local_scaled[0]),
        float(z_local_scaled[1] - z_local_scaled[0]),
    )
    origin = np.array(
        [x_local_scaled[0], y_local_scaled[0], z_local_scaled[0]],
        dtype=float,
    )

    data_min = float(np.min(volume))
    data_max = float(np.max(volume))

    for level in SNR_ISOSURFACE_LEVELS:
        level = float(level)
        if not data_min < level < data_max:
            print(
                f"Skipping SNR={level:g} isosurface; level is outside "
                "the grid range."
            )
            continue

        vertices, faces, _, _ = marching_cubes(
            volume,
            level=level,
            spacing=spacing,
        )
        vertices += origin

        figure = plt.figure(figsize=(7.0, 6.0))
        axis = figure.add_subplot(111, projection="3d")
        axis.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            faces,
            vertices[:, 2],
            linewidth=0.15,
            antialiased=True,
            alpha=0.8,
        )
        axis.set_xlabel(f"x offset ({PLOT_DISTANCE_UNIT_LABEL})")
        axis.set_ylabel(f"y offset ({PLOT_DISTANCE_UNIT_LABEL})")
        axis.set_zlabel(f"z offset ({PLOT_DISTANCE_UNIT_LABEL})")
        axis.set_title(f"SNR = {level:g} isosurface")
        figure.tight_layout()

        safe_level = str(level).replace(".", "p")
        if SAVE_PNG:
            figure.savefig(
                OUTPUT_DIRECTORY
                / f"{OUTPUT_STEM}_isosurface_snr_{safe_level}.png",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        if SAVE_PDF:
            figure.savefig(
                OUTPUT_DIRECTORY
                / f"{OUTPUT_STEM}_isosurface_snr_{safe_level}.pdf",
                bbox_inches="tight",
            )

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(figure)


def main() -> None:
    """Evaluate the grid, save data, and generate contour figures."""

    validate_configuration()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    x_km, y_km, z_km = make_grid_axes()
    positions_km = flat_grid_positions(x_km, y_km, z_km)

    total_points = positions_km.shape[0]
    approximate_snr_storage_mb = total_points * 8.0 / 1024.0**2
    print(f"Grid shape: {GRID_SHAPE}")
    print(f"Total grid points: {total_points:,}")
    print(
        "Approximate storage for one float64 grid: "
        f"{approximate_snr_storage_mb:.2f} MB"
    )
    print(f"Boresight mode: {BORESIGHT_MODE}")

    flat_outputs = evaluate_snr_grid(positions_km)
    grids = reshape_outputs(flat_outputs)
    snr_grid = grids["snr"]

    save_grid_data(x_km, y_km, z_km, grids)

    centre = np.asarray(GRID_CENTER_KM, dtype=float)
    x_local_km = x_km - centre[0]
    y_local_km = y_km - centre[1]
    z_local_km = z_km - centre[2]

    x_index = nearest_index(x_local_km, SLICE_X_OFFSET_KM)
    y_index = nearest_index(y_local_km, SLICE_Y_OFFSET_KM)
    z_index = nearest_index(z_local_km, SLICE_Z_OFFSET_KM)

    x_scaled = x_local_km / PLOT_DISTANCE_SCALE_KM
    y_scaled = y_local_km / PLOT_DISTANCE_SCALE_KM
    z_scaled = z_local_km / PLOT_DISTANCE_SCALE_KM

    print(
        "Selected contour slices (local offsets): "
        f"x={x_local_km[x_index]:.6g} km, "
        f"y={y_local_km[y_index]:.6g} km, "
        f"z={z_local_km[z_index]:.6g} km"
    )

    plot_contour_slice(
        horizontal_values=x_scaled,
        vertical_values=y_scaled,
        snr_slice=snr_grid[:, :, z_index],
        horizontal_label=f"x offset ({PLOT_DISTANCE_UNIT_LABEL})",
        vertical_label=f"y offset ({PLOT_DISTANCE_UNIT_LABEL})",
        title=(
            "Asteroid SNR in x-y plane\n"
            f"z offset = {z_local_km[z_index] / PLOT_DISTANCE_SCALE_KM:.6g} "
            f"{PLOT_DISTANCE_UNIT_LABEL}"
        ),
        output_name="xy",
    )

    plot_contour_slice(
        horizontal_values=x_scaled,
        vertical_values=z_scaled,
        snr_slice=snr_grid[:, y_index, :],
        horizontal_label=f"x offset ({PLOT_DISTANCE_UNIT_LABEL})",
        vertical_label=f"z offset ({PLOT_DISTANCE_UNIT_LABEL})",
        title=(
            "Asteroid SNR in x-z plane\n"
            f"y offset = {y_local_km[y_index] / PLOT_DISTANCE_SCALE_KM:.6g} "
            f"{PLOT_DISTANCE_UNIT_LABEL}"
        ),
        output_name="xz",
    )

    plot_contour_slice(
        horizontal_values=y_scaled,
        vertical_values=z_scaled,
        snr_slice=snr_grid[x_index, :, :],
        horizontal_label=f"y offset ({PLOT_DISTANCE_UNIT_LABEL})",
        vertical_label=f"z offset ({PLOT_DISTANCE_UNIT_LABEL})",
        title=(
            "Asteroid SNR in y-z plane\n"
            f"x offset = {x_local_km[x_index] / PLOT_DISTANCE_SCALE_KM:.6g} "
            f"{PLOT_DISTANCE_UNIT_LABEL}"
        ),
        output_name="yz",
    )

    make_isosurface_plots(
        x_local_scaled=x_scaled,
        y_local_scaled=y_scaled,
        z_local_scaled=z_scaled,
        snr_grid=snr_grid,
    )

    save_run_summary(
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        snr_grid=snr_grid,
        selected_indices=(x_index, y_index, z_index),
    )

    finite_snr = snr_grid[np.isfinite(snr_grid)]
    if finite_snr.size:
        print("SNR grid summary")
        print("----------------")
        print(f"Finite grid points: {finite_snr.size:,}")
        print(f"Minimum SNR: {np.min(finite_snr):.6g}")
        print(f"Median SNR:  {np.median(finite_snr):.6g}")
        print(f"Maximum SNR: {np.max(finite_snr):.6g}")
        for level in SNR_LINE_LEVELS:
            fraction = np.mean(finite_snr >= level)
            print(f"Fraction with SNR >= {level:g}: {fraction:.6%}")
    else:
        print("No finite SNR values were produced.")

    print(f"Outputs written to: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
