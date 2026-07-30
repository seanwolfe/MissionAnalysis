from __future__ import annotations

"""Visualize SC1's nominal SECR boresight relative to its dynamic IV zone.

This is a focused diagnostic for the initial-detection geometry. It does not
assign new boresights and does not modify the formation or mission simulation.
It answers two separate questions at every LPF orbit epoch:

1. Is the nominal boresight centre itself inside the Earth--Moon IV zone?
2. Is the centre outside, but does some part of the circular payload FOV still
   overlap the IV zone?

The script reads the configured LPF orbit and payload FOV from the mission YAML,
constructs the IV zone separately for SC1 at every sampled epoch, and creates:

* a full-period angular-clearance time history;
* angular tangent-plane snapshots at diagnostic epochs; and
* physical y-z cross-section snapshots at configurable forward distances.

Required neighboring files
--------------------------
* Mission YAML passed on the command line.
* LPF orbit CSV referenced by ``orbit_file_path`` in that YAML.
* ``earth_moon_invisibility_zone.py`` in the Python path.

Example
-------
python visualize_sc1_nominal_boresight_vs_iv_zone.py \
    sc_2_overall_orbitdetsim_config.yaml

Headless run
------------
python visualize_sc1_nominal_boresight_vs_iv_zone.py \
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
# Study options -- edit here for now
# =============================================================================

# Existing nominal initial-detection convention in SECR.
NOMINAL_BORESIGHT_SECR = np.array([-1.0, 0.0, 0.0], dtype=float)

# Fixed dynamic-IV-zone half-angle used by the mission concept.
IV_HALF_ANGLE_DEG = 12.0

# Extra required angular clearance between the FOV edge and IV-zone edge.
IV_CLEARANCE_MARGIN_DEG = 0.25

# SC1's index inside the one-period LPF slice. Rotating a complete period only
# changes where the diagnostic starts; it does not change the set of geometries.
SC1_ANCHOR_PERIOD_INDEX = 0

# Evaluate every Nth orbit row. Use 1 for every LPF sample.
SAMPLE_STRIDE = 1

# Physical planes ahead of SC1 along the nominal boresight. These figures are
# intuitive illustrations; the angular tangent-plane figure is the primary
# geometry diagnostic because it is independent of arbitrary range.
CROSS_SECTION_DISTANCES_KM = (250_000.0, 1_500_000.0)

# Number of points used to draw each cone boundary.
CONE_BOUNDARY_SAMPLES = 721

# Maximum number of diagnostic snapshot epochs.
MAX_SNAPSHOT_EPOCHS = 4

# Output folder relative to the YAML directory, unless made absolute.
OUTPUT_FOLDER = "sc1_nominal_boresight_iv_diagnostic"

SHOW_FIGURES_BY_DEFAULT = True

# Draw the raw LPF orbit diagnostics with equal geometric scaling.
RAW_ORBIT_MARKER_STRIDE = 500


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class DiagnosticData:
    times: np.ndarray
    source_row_indices: np.ndarray
    orbit_phase_deg: np.ndarray
    spacecraft_positions_secr_km: np.ndarray
    moon_positions_secr_km: np.ndarray
    iv_axes_secr: np.ndarray
    earth_los_secr: np.ndarray
    moon_los_secr: np.ndarray
    centre_separation_rad: np.ndarray
    centre_inside_raw_iv: np.ndarray
    full_fov_clearance_rad: np.ndarray
    full_fov_violation: np.ndarray
    fov_half_angle_rad: float
    iv_half_angle_rad: float
    required_centre_separation_rad: float


# =============================================================================
# Geometry helpers
# =============================================================================


def unit(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-15:
        raise ValueError(f"{name} must be a finite nonzero vector.")
    return value / norm


def fov_area_deg2_to_half_angle_rad(fov_deg2: float) -> float:
    """Match the spherical-cap equivalent circular FOV used by Spacecraft.py."""

    area_deg2 = float(fov_deg2)
    if not np.isfinite(area_deg2) or area_deg2 <= 0.0:
        raise ValueError(f"fov must be positive and finite, got {fov_deg2!r}.")

    area_sr = area_deg2 / (180.0 / np.pi) ** 2
    argument = 1.0 - area_sr / (2.0 * np.pi)
    return float(np.arccos(np.clip(argument, -1.0, 1.0)))


def angular_separation_rows(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    ref = unit(reference, "reference")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1.0e-15):
        raise ValueError("vectors contain zero-length rows.")
    values_hat = values / norms[:, None]
    return np.arccos(np.clip(values_hat @ ref, -1.0, 1.0))


def tangent_basis(reference_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return local +Y-like and +Z-like axes perpendicular to reference_axis."""

    axis = unit(reference_axis, "reference_axis")

    preferred_u = np.array([0.0, 1.0, 0.0], dtype=float)
    u = preferred_u - float(preferred_u @ axis) * axis
    if np.linalg.norm(u) <= 1.0e-12:
        preferred_u = np.array([0.0, 0.0, 1.0], dtype=float)
        u = preferred_u - float(preferred_u @ axis) * axis
    u = unit(u, "local tangent u")

    # This ordering gives +Z for the current nominal axis [-1, 0, 0].
    v = unit(np.cross(u, axis), "local tangent v")
    return u, v


def cone_boundary(axis: np.ndarray, half_angle_rad: float, samples: int) -> np.ndarray:
    """Sample unit directions on a circular cone boundary."""

    centre = unit(axis, "cone axis")
    u, v = tangent_basis(centre)
    phi = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=True)
    boundary = (
        np.cos(half_angle_rad) * centre[None, :]
        + np.sin(half_angle_rad)
        * (
            np.cos(phi)[:, None] * u[None, :]
            + np.sin(phi)[:, None] * v[None, :]
        )
    )
    return boundary / np.linalg.norm(boundary, axis=1)[:, None]


def tangent_plane_projection_deg(
    directions: np.ndarray,
    reference_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gnomonic-like signed angular coordinates around reference_axis.

    The coordinates are reported as separate signed angular offsets along the
    local tangent axes. They are especially intuitive for the small angular
    fields considered here.
    """

    values = np.asarray(directions, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, 3)

    values = values / np.linalg.norm(values, axis=1)[:, None]
    centre = unit(reference_axis, "reference_axis")
    u, v = tangent_basis(centre)

    forward = values @ centre
    horizontal = values @ u
    vertical = values @ v

    x_deg = np.rad2deg(np.arctan2(horizontal, forward))
    y_deg = np.rad2deg(np.arctan2(vertical, forward))
    return x_deg, y_deg, forward


def plane_intersections_km(
    directions: np.ndarray,
    reference_axis: np.ndarray,
    forward_distance_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect rays with a plane normal to reference_axis at +distance."""

    values = np.asarray(directions, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, 3)

    values = values / np.linalg.norm(values, axis=1)[:, None]
    centre = unit(reference_axis, "reference_axis")
    u, v = tangent_basis(centre)

    denominator = values @ centre
    valid = denominator > 1.0e-12

    scale = np.full(values.shape[0], np.nan, dtype=float)
    scale[valid] = float(forward_distance_km) / denominator[valid]
    points = values * scale[:, None]

    y_km = points @ u
    z_km = points @ v
    return y_km, z_km, valid


# =============================================================================
# Configuration and LPF data
# =============================================================================


def resolve_relative_path(value: str | Path, config_path: Path) -> Path:
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


def load_lpf_one_period(config: dict, config_path: Path) -> pd.DataFrame:
    orbit_path = resolve_relative_path(config["orbit_file_path"], config_path)
    if not orbit_path.exists():
        raise FileNotFoundError(
            f"Configured LPF orbit file does not exist: {orbit_path}\n"
            "Place the CSV beside the YAML or update orbit_file_path."
        )

    column_names = config.get("orbit_column_names")
    orbit = pd.read_csv(
        orbit_path,
        sep=",",
        header=0,
        names=column_names,
    )

    start = int(config["quasi_halo_start"])
    end = int(config["quasi_halo_one_period_end"])
    if not 0 <= start < end <= len(orbit):
        raise ValueError(
            "Invalid one-period LPF slice: "
            f"start={start}, end={end}, orbit rows={len(orbit)}."
        )

    period = orbit.iloc[start:end].copy().reset_index(drop=False)
    period = period.rename(columns={"index": "SOURCE_ROW_INDEX"})

    required_columns = [
        "SUN_EARTH_CO_X_(km)",
        "SUN_EARTH_CO_Y_(km)",
        "SUN_EARTH_CO_Z_(km)",
        "MOON_SUN_EARTH_CO_X_(km)",
        "MOON_SUN_EARTH_CO_Y_(km)",
        "MOON_SUN_EARTH_CO_Z_(km)",
    ]
    missing = [column for column in required_columns if column not in period.columns]
    if missing:
        raise KeyError(f"LPF orbit is missing required SECR columns: {missing}")

    if "Time" in period.columns:
        period["Time"] = pd.to_datetime(period["Time"], errors="coerce")

    anchor = int(SC1_ANCHOR_PERIOD_INDEX) % len(period)
    if anchor:
        period = pd.concat(
            [period.iloc[anchor:], period.iloc[:anchor]],
            ignore_index=True,
        )

    stride = int(SAMPLE_STRIDE)
    if stride <= 0:
        raise ValueError("SAMPLE_STRIDE must be a positive integer.")

    sampled = period.iloc[::stride].copy().reset_index(drop=True)
    if len(sampled) < 2:
        raise ValueError(
            "Too few sampled LPF epochs. Reduce SAMPLE_STRIDE or verify the orbit slice."
        )
    return sampled


def build_diagnostic_data(
    config: dict,
    config_path: Path,
    *,
    anchor_period_index: int | None = None,
    stride_override: int | None = None,
) -> DiagnosticData:
    global SC1_ANCHOR_PERIOD_INDEX, SAMPLE_STRIDE

    old_anchor = SC1_ANCHOR_PERIOD_INDEX
    old_stride = SAMPLE_STRIDE
    try:
        if anchor_period_index is not None:
            SC1_ANCHOR_PERIOD_INDEX = int(anchor_period_index)
        if stride_override is not None:
            SAMPLE_STRIDE = int(stride_override)
        orbit = load_lpf_one_period(config, config_path)
    finally:
        SC1_ANCHOR_PERIOD_INDEX = old_anchor
        SAMPLE_STRIDE = old_stride

    spacecraft_positions = orbit.loc[
        :,
        [
            "SUN_EARTH_CO_X_(km)",
            "SUN_EARTH_CO_Y_(km)",
            "SUN_EARTH_CO_Z_(km)",
        ],
    ].to_numpy(dtype=float)

    moon_positions = orbit.loc[
        :,
        [
            "MOON_SUN_EARTH_CO_X_(km)",
            "MOON_SUN_EARTH_CO_Y_(km)",
            "MOON_SUN_EARTH_CO_Z_(km)",
        ],
    ].to_numpy(dtype=float)

    earth_positions = np.zeros_like(spacecraft_positions)
    zone = compute_earth_moon_invisibility_zone_batch(
        spacecraft_position=spacecraft_positions,
        earth_position=earth_positions,
        moon_position=moon_positions,
        half_angle_deg=IV_HALF_ANGLE_DEG,
    )

    nominal = unit(NOMINAL_BORESIGHT_SECR, "NOMINAL_BORESIGHT_SECR")
    centre_separation = angular_separation_rows(zone.axis_geo_eme, nominal)

    fov_half_angle = fov_area_deg2_to_half_angle_rad(config["fov"])
    iv_half_angle = float(zone.half_angle_rad)
    margin = float(np.deg2rad(IV_CLEARANCE_MARGIN_DEG))
    required_centre_separation = iv_half_angle + fov_half_angle + margin

    centre_inside_raw_iv = centre_separation <= iv_half_angle
    full_fov_clearance = centre_separation - required_centre_separation
    full_fov_violation = full_fov_clearance < 0.0

    sample_count = len(orbit)
    orbit_phase_deg = np.linspace(0.0, 360.0, sample_count, endpoint=False)

    if "Time" in orbit.columns:
        times = orbit["Time"].to_numpy()
    else:
        times = np.arange(sample_count)

    return DiagnosticData(
        times=times,
        source_row_indices=orbit["SOURCE_ROW_INDEX"].to_numpy(dtype=int),
        orbit_phase_deg=orbit_phase_deg,
        spacecraft_positions_secr_km=spacecraft_positions,
        moon_positions_secr_km=moon_positions,
        iv_axes_secr=np.asarray(zone.axis_geo_eme, dtype=float),
        earth_los_secr=np.asarray(zone.earth_los_geo_eme, dtype=float),
        moon_los_secr=np.asarray(zone.moon_los_geo_eme, dtype=float),
        centre_separation_rad=centre_separation,
        centre_inside_raw_iv=centre_inside_raw_iv,
        full_fov_clearance_rad=full_fov_clearance,
        full_fov_violation=full_fov_violation,
        fov_half_angle_rad=fov_half_angle,
        iv_half_angle_rad=iv_half_angle,
        required_centre_separation_rad=required_centre_separation,
    )


# =============================================================================
# Diagnostic epoch selection
# =============================================================================


def select_snapshot_indices(data: DiagnosticData) -> list[int]:
    selected: list[int] = []

    def add(index: int | None) -> None:
        if index is None:
            return
        value = int(index)
        if value not in selected:
            selected.append(value)

    # Most negative FOV-edge clearance: clearest picture of the worst overlap.
    add(int(np.argmin(data.full_fov_clearance_rad)))

    violation_indices = np.flatnonzero(data.full_fov_violation)
    if violation_indices.size:
        add(int(violation_indices[0]))

    # Safest epoch closest to the boundary, useful for seeing just-touching geometry.
    safe_indices = np.flatnonzero(~data.full_fov_violation)
    if safe_indices.size:
        closest_safe_local = int(
            np.argmin(data.full_fov_clearance_rad[safe_indices])
        )
        add(int(safe_indices[closest_safe_local]))

    # Best-clearance epoch for contrast.
    add(int(np.argmax(data.full_fov_clearance_rad)))

    return selected[: int(MAX_SNAPSHOT_EPOCHS)]


def epoch_label(data: DiagnosticData, index: int) -> str:
    time_value = data.times[index]
    if isinstance(time_value, np.datetime64) and not np.isnat(time_value):
        time_text = pd.Timestamp(time_value).isoformat(sep=" ")
    else:
        time_text = str(time_value)

    clearance_deg = float(np.rad2deg(data.full_fov_clearance_rad[index]))
    status = "VIOLATION" if clearance_deg < 0.0 else "clear"
    return (
        f"phase {data.orbit_phase_deg[index]:.1f} deg | {status}\n"
        f"edge clearance {clearance_deg:+.3f} deg | {time_text}"
    )


# =============================================================================
# Output tables and plots
# =============================================================================


def save_diagnostic_table(data: DiagnosticData, output_dir: Path) -> Path:
    table = pd.DataFrame(
        {
            "sample_index": np.arange(len(data.orbit_phase_deg), dtype=int),
            "source_row_index": data.source_row_indices,
            "time": data.times,
            "orbit_phase_deg": data.orbit_phase_deg,
            "sc_x_secr_km": data.spacecraft_positions_secr_km[:, 0],
            "sc_y_secr_km": data.spacecraft_positions_secr_km[:, 1],
            "sc_z_secr_km": data.spacecraft_positions_secr_km[:, 2],
            "moon_x_secr_km": data.moon_positions_secr_km[:, 0],
            "moon_y_secr_km": data.moon_positions_secr_km[:, 1],
            "moon_z_secr_km": data.moon_positions_secr_km[:, 2],
            "iv_axis_x_secr": data.iv_axes_secr[:, 0],
            "iv_axis_y_secr": data.iv_axes_secr[:, 1],
            "iv_axis_z_secr": data.iv_axes_secr[:, 2],
            "centre_separation_deg": np.rad2deg(data.centre_separation_rad),
            "iv_half_angle_deg": np.rad2deg(data.iv_half_angle_rad),
            "fov_half_angle_deg": np.rad2deg(data.fov_half_angle_rad),
            "iv_clearance_margin_deg": IV_CLEARANCE_MARGIN_DEG,
            "required_centre_separation_deg": np.rad2deg(
                data.required_centre_separation_rad
            ),
            "centre_inside_raw_iv": data.centre_inside_raw_iv,
            "full_fov_clearance_deg": np.rad2deg(data.full_fov_clearance_rad),
            "full_fov_violation": data.full_fov_violation,
        }
    )
    path = output_dir / "sc1_nominal_boresight_iv_diagnostic.csv"
    table.to_csv(path, index=False)
    return path


def plot_clearance_history(
    data: DiagnosticData,
    snapshot_indices: list[int],
    output_dir: Path,
) -> Path:
    x = data.orbit_phase_deg
    separation_deg = np.rad2deg(data.centre_separation_rad)
    iv_deg = float(np.rad2deg(data.iv_half_angle_rad))
    required_deg = float(np.rad2deg(data.required_centre_separation_rad))

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.plot(x, separation_deg, linewidth=1.5, label="Nominal boresight to IV-axis separation")
    ax.axhline(iv_deg, linestyle="--", linewidth=1.2, label="Raw IV half-angle")
    ax.axhline(
        required_deg,
        linestyle=":",
        linewidth=1.5,
        label="Required for complete FOV clearance",
    )

    ax.fill_between(
        x,
        separation_deg,
        required_deg,
        where=data.full_fov_violation,
        interpolate=True,
        alpha=0.25,
        label="Complete-FOV violation",
    )

    raw_inside = data.centre_inside_raw_iv
    if np.any(raw_inside):
        ax.scatter(
            x[raw_inside],
            separation_deg[raw_inside],
            marker="x",
            s=24,
            label="Nominal centre inside raw IV zone",
        )

    for index in snapshot_indices:
        ax.scatter(x[index], separation_deg[index], s=34, zorder=5)
        ax.annotate(
            str(index),
            (x[index], separation_deg[index]),
            xytext=(4, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("SC1 phase through one LPF halo period (deg)")
    ax.set_ylabel("Angular separation (deg)")
    ax.set_xlim(0.0, 360.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    path = output_dir / "sc1_nominal_boresight_iv_clearance_history.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    return path


def plot_clearance_margin_history(data: DiagnosticData, output_dir: Path) -> Path:
    clearance_deg = np.rad2deg(data.full_fov_clearance_rad)

    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.plot(data.orbit_phase_deg, clearance_deg, linewidth=1.5)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.fill_between(
        data.orbit_phase_deg,
        clearance_deg,
        0.0,
        where=clearance_deg < 0.0,
        interpolate=True,
        alpha=0.25,
    )
    ax.set_xlabel("SC1 phase through one LPF halo period (deg)")
    ax.set_ylabel("Complete-FOV edge clearance (deg)")
    ax.set_xlim(0.0, 360.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / "sc1_nominal_boresight_iv_edge_clearance.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    return path


def plot_angular_snapshots(
    data: DiagnosticData,
    snapshot_indices: list[int],
    output_dir: Path,
) -> Path:
    nominal = unit(NOMINAL_BORESIGHT_SECR, "NOMINAL_BORESIGHT_SECR")
    fov_boundary = cone_boundary(
        nominal,
        data.fov_half_angle_rad,
        CONE_BOUNDARY_SAMPLES,
    )
    fov_x, fov_y, _ = tangent_plane_projection_deg(fov_boundary, nominal)

    panel_data: list[dict] = []
    all_x = [fov_x]
    all_y = [fov_y]

    for index in snapshot_indices:
        iv_axis = data.iv_axes_secr[index]
        iv_boundary = cone_boundary(
            iv_axis,
            data.iv_half_angle_rad,
            CONE_BOUNDARY_SAMPLES,
        )
        iv_x, iv_y, iv_forward = tangent_plane_projection_deg(iv_boundary, nominal)

        axis_x, axis_y, axis_forward = tangent_plane_projection_deg(iv_axis, nominal)
        earth_x, earth_y, earth_forward = tangent_plane_projection_deg(
            data.earth_los_secr[index], nominal
        )
        moon_x, moon_y, moon_forward = tangent_plane_projection_deg(
            data.moon_los_secr[index], nominal
        )

        panel_data.append(
            {
                "index": index,
                "iv_x": iv_x,
                "iv_y": iv_y,
                "iv_forward": iv_forward,
                "axis_x": axis_x,
                "axis_y": axis_y,
                "axis_forward": axis_forward,
                "earth_x": earth_x,
                "earth_y": earth_y,
                "earth_forward": earth_forward,
                "moon_x": moon_x,
                "moon_y": moon_y,
                "moon_forward": moon_forward,
            }
        )

        visible = iv_forward > 0.0
        if np.any(visible):
            all_x.append(iv_x[visible])
            all_y.append(iv_y[visible])

    combined_x = np.concatenate(all_x)
    combined_y = np.concatenate(all_y)
    finite = np.isfinite(combined_x) & np.isfinite(combined_y)
    if not np.any(finite):
        raise RuntimeError("No finite angular snapshot geometry was produced.")

    x_min, x_max = np.min(combined_x[finite]), np.max(combined_x[finite])
    y_min, y_max = np.min(combined_y[finite]), np.max(combined_y[finite])
    span = max(x_max - x_min, y_max - y_min, 1.0)
    padding = 0.10 * span

    count = len(panel_data)
    ncols = min(2, count)
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.3 * ncols, 5.8 * nrows),
        squeeze=False,
    )

    for panel_index, panel in enumerate(panel_data):
        ax = axes.flat[panel_index]
        index = panel["index"]

        ax.fill(fov_x, fov_y, alpha=0.18, label="Nominal FOV")
        ax.plot(fov_x, fov_y, linewidth=1.5)

        visible = panel["iv_forward"] > 0.0
        if np.count_nonzero(visible) >= 3:
            ax.fill(
                panel["iv_x"][visible],
                panel["iv_y"][visible],
                alpha=0.22,
                label="Dynamic IV zone",
            )
            ax.plot(
                panel["iv_x"][visible],
                panel["iv_y"][visible],
                linewidth=1.5,
            )
        else:
            ax.text(
                0.03,
                0.96,
                "IV cone does not intersect the forward nominal hemisphere",
                transform=ax.transAxes,
                va="top",
            )

        ax.scatter(0.0, 0.0, marker="+", s=90, label="Nominal boresight")

        if panel["axis_forward"][0] > 0.0:
            ax.scatter(panel["axis_x"][0], panel["axis_y"][0], marker="x", s=65, label="IV axis")
        if panel["earth_forward"][0] > 0.0:
            ax.scatter(panel["earth_x"][0], panel["earth_y"][0], marker="o", s=38, label="Earth LOS")
        if panel["moon_forward"][0] > 0.0:
            ax.scatter(panel["moon_x"][0], panel["moon_y"][0], marker="s", s=34, label="Moon LOS")

        ax.set_title(f"Snapshot {index}: {epoch_label(data, index)}")
        ax.set_xlabel("Local +Y angular offset from nominal (deg)")
        ax.set_ylabel("Local +Z angular offset from nominal (deg)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")

    for unused in range(count, nrows * ncols):
        axes.flat[unused].set_visible(False)

    fig.tight_layout()
    path = output_dir / "sc1_nominal_boresight_iv_angular_snapshots.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    return path


def plot_physical_cross_sections(
    data: DiagnosticData,
    snapshot_indices: list[int],
    output_dir: Path,
    distance_km: float,
) -> Path:
    nominal = unit(NOMINAL_BORESIGHT_SECR, "NOMINAL_BORESIGHT_SECR")
    fov_boundary = cone_boundary(
        nominal,
        data.fov_half_angle_rad,
        CONE_BOUNDARY_SAMPLES,
    )
    fov_y, fov_z, fov_valid = plane_intersections_km(
        fov_boundary,
        nominal,
        distance_km,
    )

    panel_data: list[dict] = []
    all_y = [fov_y[fov_valid]]
    all_z = [fov_z[fov_valid]]

    for index in snapshot_indices:
        iv_axis = data.iv_axes_secr[index]
        iv_boundary = cone_boundary(
            iv_axis,
            data.iv_half_angle_rad,
            CONE_BOUNDARY_SAMPLES,
        )
        iv_y, iv_z, iv_valid = plane_intersections_km(
            iv_boundary,
            nominal,
            distance_km,
        )

        axis_y, axis_z, axis_valid = plane_intersections_km(
            iv_axis,
            nominal,
            distance_km,
        )
        earth_y, earth_z, earth_valid = plane_intersections_km(
            data.earth_los_secr[index],
            nominal,
            distance_km,
        )
        moon_y, moon_z, moon_valid = plane_intersections_km(
            data.moon_los_secr[index],
            nominal,
            distance_km,
        )

        panel_data.append(
            {
                "index": index,
                "iv_y": iv_y,
                "iv_z": iv_z,
                "iv_valid": iv_valid,
                "axis_y": axis_y,
                "axis_z": axis_z,
                "axis_valid": axis_valid,
                "earth_y": earth_y,
                "earth_z": earth_z,
                "earth_valid": earth_valid,
                "moon_y": moon_y,
                "moon_z": moon_z,
                "moon_valid": moon_valid,
            }
        )
        if np.any(iv_valid):
            all_y.append(iv_y[iv_valid])
            all_z.append(iv_z[iv_valid])

    combined_y = np.concatenate(all_y)
    combined_z = np.concatenate(all_z)
    finite = np.isfinite(combined_y) & np.isfinite(combined_z)
    if not np.any(finite):
        raise RuntimeError("No finite cross-section geometry was produced.")

    y_min, y_max = np.min(combined_y[finite]), np.max(combined_y[finite])
    z_min, z_max = np.min(combined_z[finite]), np.max(combined_z[finite])
    span = max(y_max - y_min, z_max - z_min, 1.0)
    padding = 0.10 * span

    count = len(panel_data)
    ncols = min(2, count)
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.3 * ncols, 5.8 * nrows),
        squeeze=False,
    )

    for panel_index, panel in enumerate(panel_data):
        ax = axes.flat[panel_index]
        index = panel["index"]

        ax.fill(fov_y[fov_valid], fov_z[fov_valid], alpha=0.18, label="Nominal FOV")
        ax.plot(fov_y[fov_valid], fov_z[fov_valid], linewidth=1.5)

        if np.count_nonzero(panel["iv_valid"]) >= 3:
            ax.fill(
                panel["iv_y"][panel["iv_valid"]],
                panel["iv_z"][panel["iv_valid"]],
                alpha=0.22,
                label="Dynamic IV zone",
            )
            ax.plot(
                panel["iv_y"][panel["iv_valid"]],
                panel["iv_z"][panel["iv_valid"]],
                linewidth=1.5,
            )

        ax.scatter(0.0, 0.0, marker="+", s=90, label="Nominal boresight")
        if panel["axis_valid"][0]:
            ax.scatter(panel["axis_y"][0], panel["axis_z"][0], marker="x", s=65, label="IV axis")
        if panel["earth_valid"][0]:
            ax.scatter(panel["earth_y"][0], panel["earth_z"][0], marker="o", s=38, label="Earth LOS")
        if panel["moon_valid"][0]:
            ax.scatter(panel["moon_y"][0], panel["moon_z"][0], marker="s", s=34, label="Moon LOS")

        ax.set_title(f"Snapshot {index}: {epoch_label(data, index)}")
        ax.set_xlabel("Local +Y at cross section (km)")
        ax.set_ylabel("Local +Z at cross section (km)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(y_min - padding, y_max + padding)
        ax.set_ylim(z_min - padding, z_max + padding)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")

    for unused in range(count, nrows * ncols):
        axes.flat[unused].set_visible(False)

    fig.tight_layout()
    distance_tag = f"{distance_km:g}".replace(".", "p")
    path = output_dir / f"sc1_nominal_boresight_iv_yz_cross_section_{distance_tag}_km.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    return path




def _set_3d_equal_limits(ax, arrays: list[np.ndarray], padding_fraction: float = 0.06) -> None:
    """Apply equal x/y/z scale to a Matplotlib 3D axis."""

    points = np.vstack([np.asarray(values, dtype=float).reshape(-1, 3) for values in arrays])
    minimum = np.nanmin(points, axis=0)
    maximum = np.nanmax(points, axis=0)
    centre = 0.5 * (minimum + maximum)
    half_range = 0.5 * float(np.max(maximum - minimum))
    if not np.isfinite(half_range) or half_range <= 0.0:
        half_range = 1.0
    half_range *= 1.0 + float(padding_fraction)

    ax.set_xlim(centre[0] - half_range, centre[0] + half_range)
    ax.set_ylim(centre[1] - half_range, centre[1] + half_range)
    ax.set_zlim(centre[2] - half_range, centre[2] + half_range)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def plot_raw_orbit_3d(
    data: DiagnosticData,
    snapshot_indices: list[int],
    output_dir: Path,
) -> Path:
    """Plot the raw geocentric SECR trajectories over the analyzed period."""

    sc = data.spacecraft_positions_secr_km
    moon = data.moon_positions_secr_km
    earth = np.zeros((1, 3), dtype=float)

    fig = plt.figure(figsize=(9.0, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sc[:, 0], sc[:, 1], sc[:, 2], linewidth=1.5, label="SC1 LPF trajectory")
    ax.plot(moon[:, 0], moon[:, 1], moon[:, 2], linewidth=1.1, label="Moon trajectory")
    ax.scatter(0.0, 0.0, 0.0, s=70, marker="o", label="Earth")

    if snapshot_indices:
        indices = np.asarray(snapshot_indices, dtype=int)
        ax.scatter(
            sc[indices, 0], sc[indices, 1], sc[indices, 2],
            s=42, marker="x", label="Selected diagnostic epochs",
        )
        for index in indices:
            ax.text(sc[index, 0], sc[index, 1], sc[index, 2], f" {index}", fontsize=8)

    marker_stride = max(1, int(RAW_ORBIT_MARKER_STRIDE))
    ax.scatter(
        sc[::marker_stride, 0], sc[::marker_stride, 1], sc[::marker_stride, 2],
        s=8, alpha=0.45,
    )

    ax.set_xlabel("SECR x (km)")
    ax.set_ylabel("SECR y (km)")
    ax.set_zlabel("SECR z (km)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _set_3d_equal_limits(ax, [sc, moon, earth])
    fig.tight_layout()

    path = output_dir / "sc1_raw_lpf_earth_moon_orbits_3d.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    return path


def plot_raw_orbit_yz(
    data: DiagnosticData,
    snapshot_indices: list[int],
    output_dir: Path,
) -> Path:
    """Plot the raw geocentric SECR y-z projection with equal axis scaling."""

    sc = data.spacecraft_positions_secr_km
    moon = data.moon_positions_secr_km

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.plot(sc[:, 1], sc[:, 2], linewidth=1.5, label="SC1 LPF trajectory")
    ax.plot(moon[:, 1], moon[:, 2], linewidth=1.1, label="Moon trajectory")
    ax.scatter(0.0, 0.0, s=70, marker="o", label="Earth")

    if snapshot_indices:
        indices = np.asarray(snapshot_indices, dtype=int)
        ax.scatter(
            sc[indices, 1], sc[indices, 2],
            s=42, marker="x", label="Selected diagnostic epochs",
        )
        for index in indices:
            ax.annotate(
                str(index),
                (sc[index, 1], sc[index, 2]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("SECR y (km)")
    ax.set_ylabel("SECR z (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    path = output_dir / "sc1_raw_lpf_earth_moon_orbits_yz.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    return path


def raw_orbit_metrics(data: DiagnosticData) -> dict:
    """Return dimensional checks useful for catching frame/column mistakes."""

    sc = data.spacecraft_positions_secr_km
    moon = data.moon_positions_secr_km
    earth_range = np.linalg.norm(sc, axis=1)
    moon_range = np.linalg.norm(moon - sc, axis=1)

    return {
        "sc_x_min_km": float(np.min(sc[:, 0])),
        "sc_x_max_km": float(np.max(sc[:, 0])),
        "sc_y_min_km": float(np.min(sc[:, 1])),
        "sc_y_max_km": float(np.max(sc[:, 1])),
        "sc_z_min_km": float(np.min(sc[:, 2])),
        "sc_z_max_km": float(np.max(sc[:, 2])),
        "sc_y_peak_to_peak_km": float(np.ptp(sc[:, 1])),
        "sc_z_peak_to_peak_km": float(np.ptp(sc[:, 2])),
        "moon_y_peak_to_peak_km": float(np.ptp(moon[:, 1])),
        "moon_z_peak_to_peak_km": float(np.ptp(moon[:, 2])),
        "minimum_sc_earth_range_km": float(np.min(earth_range)),
        "maximum_sc_earth_range_km": float(np.max(earth_range)),
        "minimum_sc_moon_range_km": float(np.min(moon_range)),
        "maximum_sc_moon_range_km": float(np.max(moon_range)),
    }

# =============================================================================
# Console summary
# =============================================================================


def build_summary(data: DiagnosticData, snapshot_indices: list[int]) -> dict:
    full_count = int(np.count_nonzero(data.full_fov_violation))
    raw_count = int(np.count_nonzero(data.centre_inside_raw_iv))
    total = int(len(data.orbit_phase_deg))
    worst_index = int(np.argmin(data.full_fov_clearance_rad))

    return {
        "sampled_epoch_count": total,
        "nominal_boresight_secr": unit(
            NOMINAL_BORESIGHT_SECR, "NOMINAL_BORESIGHT_SECR"
        ).tolist(),
        "fov_area_deg2": None,
        "fov_half_angle_deg": float(np.rad2deg(data.fov_half_angle_rad)),
        "iv_half_angle_deg": float(np.rad2deg(data.iv_half_angle_rad)),
        "iv_clearance_margin_deg": float(IV_CLEARANCE_MARGIN_DEG),
        "required_centre_separation_deg": float(
            np.rad2deg(data.required_centre_separation_rad)
        ),
        "raw_centre_inside_iv_count": raw_count,
        "raw_centre_inside_iv_fraction": raw_count / total,
        "complete_fov_violation_count": full_count,
        "complete_fov_violation_fraction": full_count / total,
        "worst_sample_index": worst_index,
        "worst_source_row_index": int(data.source_row_indices[worst_index]),
        "worst_orbit_phase_deg": float(data.orbit_phase_deg[worst_index]),
        "worst_centre_separation_deg": float(
            np.rad2deg(data.centre_separation_rad[worst_index])
        ),
        "worst_complete_fov_clearance_deg": float(
            np.rad2deg(data.full_fov_clearance_rad[worst_index])
        ),
        "snapshot_indices": [int(value) for value in snapshot_indices],
    }


def print_interpretation(summary: dict) -> None:
    raw = summary["raw_centre_inside_iv_count"]
    full = summary["complete_fov_violation_count"]
    total = summary["sampled_epoch_count"]

    print("\nSC1 nominal-boresight / IV-zone diagnostic")
    print("-" * 52)
    print(f"Sampled epochs: {total}")
    print(
        "Equivalent circular FOV half-angle: "
        f"{summary['fov_half_angle_deg']:.6f} deg"
    )
    print(f"IV-zone half-angle: {summary['iv_half_angle_deg']:.6f} deg")
    print(
        "Required nominal-centre separation for complete FOV clearance: "
        f"{summary['required_centre_separation_deg']:.6f} deg"
    )
    print(f"Nominal centre inside raw IV zone: {raw}/{total}")
    print(f"Complete nominal FOV violates IV clearance: {full}/{total}")
    print(
        "Worst complete-FOV edge clearance: "
        f"{summary['worst_complete_fov_clearance_deg']:+.6f} deg "
        f"at phase {summary['worst_orbit_phase_deg']:.3f} deg"
    )

    if raw == 0 and full > 0:
        print(
            "Interpretation: the nominal boresight centre never enters the raw "
            "IV zone, but the edge of its finite FOV overlaps the IV zone at "
            "some epochs."
        )
    elif raw > 0:
        print(
            "Interpretation: at some epochs the nominal boresight centre itself "
            "lies inside the raw IV zone; these are stronger violations than "
            "edge-only overlap."
        )
    else:
        print(
            "Interpretation: the complete nominal FOV clears the IV zone at all "
            "sampled epochs under the configured margin."
        )


# =============================================================================
# Main
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Mission YAML path.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening interactive windows.",
    )
    parser.add_argument(
        "--anchor-period-index",
        type=int,
        default=None,
        help="Override SC1_ANCHOR_PERIOD_INDEX for this run.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override SAMPLE_STRIDE for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config_path = args.config.expanduser().resolve()
    config = load_configuration(config_path)

    data = build_diagnostic_data(
        config,
        config_path,
        anchor_period_index=args.anchor_period_index,
        stride_override=args.stride,
    )
    snapshot_indices = select_snapshot_indices(data)

    output_dir = Path(OUTPUT_FOLDER).expanduser()
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = save_diagnostic_table(data, output_dir)
    history_path = plot_clearance_history(data, snapshot_indices, output_dir)
    margin_path = plot_clearance_margin_history(data, output_dir)
    angular_path = plot_angular_snapshots(data, snapshot_indices, output_dir)
    raw_orbit_3d_path = plot_raw_orbit_3d(data, snapshot_indices, output_dir)
    raw_orbit_yz_path = plot_raw_orbit_yz(data, snapshot_indices, output_dir)

    cross_section_paths = [
        plot_physical_cross_sections(data, snapshot_indices, output_dir, distance)
        for distance in CROSS_SECTION_DISTANCES_KM
    ]

    summary = build_summary(data, snapshot_indices)
    summary["fov_area_deg2"] = float(config["fov"])
    summary["raw_orbit_metrics"] = raw_orbit_metrics(data)
    summary["output_files"] = {
        "diagnostic_csv": str(table_path),
        "clearance_history_svg": str(history_path),
        "edge_clearance_svg": str(margin_path),
        "angular_snapshots_svg": str(angular_path),
        "raw_orbit_3d_svg": str(raw_orbit_3d_path),
        "raw_orbit_yz_svg": str(raw_orbit_yz_path),
        "cross_section_svgs": [str(path) for path in cross_section_paths],
    }

    summary_path = output_dir / "sc1_nominal_boresight_iv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_interpretation(summary)
    metrics = summary["raw_orbit_metrics"]
    print("\nRaw geocentric SECR orbit checks")
    print("-" * 52)
    print(
        f"SC1 y peak-to-peak: {metrics['sc_y_peak_to_peak_km']:,.3f} km "
        f"(semi-amplitude approximately {0.5 * metrics['sc_y_peak_to_peak_km']:,.3f} km)"
    )
    print(
        f"SC1 z peak-to-peak: {metrics['sc_z_peak_to_peak_km']:,.3f} km "
        f"(semi-amplitude approximately {0.5 * metrics['sc_z_peak_to_peak_km']:,.3f} km)"
    )
    print(
        "SC1--Earth range: "
        f"{metrics['minimum_sc_earth_range_km']:,.3f} to "
        f"{metrics['maximum_sc_earth_range_km']:,.3f} km"
    )
    print(
        "SC1--Moon range: "
        f"{metrics['minimum_sc_moon_range_km']:,.3f} to "
        f"{metrics['maximum_sc_moon_range_km']:,.3f} km"
    )
    print(f"\nOutputs written to: {output_dir}")
    for name, value in summary["output_files"].items():
        print(f"  {name}: {value}")
    print(f"  summary_json: {summary_path}")

    show = SHOW_FIGURES_BY_DEFAULT and not args.no_show
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
