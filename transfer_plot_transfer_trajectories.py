"""Plot geocentric transfer trajectories and impulsive manoeuvres.

PyCharm usage
-------------
1. Edit the values in the ``USER CONFIGURATION`` section below.
2. Open this file in PyCharm.
3. Press the green Run button.

The script automatically distinguishes trajectory CSVs from manoeuvre CSVs by
examining their column names, pairs files by their shared filename stem, and
plots all selected cases on the same axes. Files containing ``best_dv`` and
``median_dv`` are included by default.

Required trajectory columns
---------------------------
    jdtdb, utc, t_days_since_departure,
    x_geo_j2000_km, y_geo_j2000_km, z_geo_j2000_km,
    vx_geo_j2000_kms, vy_geo_j2000_kms, vz_geo_j2000_kms,
    r_geo_km, v_geo_kms, C3_geo_km2_s2

Required manoeuvre columns
--------------------------
    name, label, fraction, jdtdb, utc,
    dv_x_kms, dv_y_kms, dv_z_kms, dv_mag_mps

Notes
-----
* ``fraction`` is interpreted as elapsed-transfer fraction from 0 to 1.
* Manoeuvre annotations use the ``name`` column, not ``label``.
* Legend entries are shortened to the best/median total Delta-v.
* The Moon is queried in the Earth-centred J2000 frame using SPICE with no
  aberration correction, matching the geocentric J2000 trajectory columns.
* At least one SPK kernel covering Earth and the Moon is needed to draw the
  Moon. A SPICE meta-kernel may also be supplied.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


TRAJECTORY_REQUIRED = {
    "jdtdb",
    "x_geo_j2000_km",
    "y_geo_j2000_km",
    "z_geo_j2000_km",
}

MANOEUVRE_REQUIRED = {
    "fraction",
    "dv_mag_mps",
}

TRAJECTORY_ROLE_WORDS = {
    "trajectory",
    "trajectories",
    "traj",
    "state",
    "states",
    "ephemeris",
    "ephem",
    "history",
}

MANOEUVRE_ROLE_WORDS = {
    "maneuver",
    "maneuvers",
    "manoeuvre",
    "manoeuvres",
    "burn",
    "burns",
    "impulse",
    "impulses",
    "dvtable",
    "dv_table",
}

GENERIC_WORDS = {
    "output",
    "outputs",
    "result",
    "results",
    "data",
    "csv",
}

MARKERS = ["s", "^", "D", "P", "v", "X", "*", "<", ">", "h", "p", "8"]
LINESTYLES = {
    "best_dv": "-",
    "median_dv": "--",
    "other": "-.",
}

J2000_JD_TDB = 2451545.0
SECONDS_PER_DAY = 86400.0


# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Edit this section, then press the green Run button in PyCharm.
# For Windows paths, keep the leading r before the quotation mark.

# Folder containing trajectory CSVs.
TRAJECTORY_DIR = Path(r"C:\Users\seanw\Desktop\MissionAnalysis")

# Folder containing manoeuvre CSVs. Use the same folder as TRAJECTORY_DIR when
# both CSV types are stored together.
MANOEUVRE_DIR = TRAJECTORY_DIR

# Only files containing at least one of these terms are considered. The default
# selects the best-Delta-v and median-Delta-v cases. Use [] to include every
# recognized trajectory and manoeuvre CSV in the configured folders.
NAME_FILTERS = ["best_dv", "median_dv"]

# Search through subfolders as well.
RECURSIVE_SEARCH = False

# SPICE kernels or a meta-kernel. Multiple entries may be supplied. Leave this
# list empty to omit the Moon. The kernel set must cover Earth and the Moon over
# all trajectory dates.
SPICE_KERNELS = [
    Path(r"C:\Users\seanw\Desktop\MissionAnalysis\naif0012.tls"),
    Path(r"C:\Users\seanw\Desktop\MissionAnalysis\de430.bsp"),
]
# Number of epochs used to draw the Moon path.
MOON_SAMPLES = 500

# Any combination of: "3d", "xy", "xz", and "yz".
VIEWS = ["3d"]

# Saved figure settings.
OUTPUT_DIR = Path(__file__).resolve().parent / "trajectory_plots"
OUTPUT_FORMATS = ["png"]       # Examples: ["png", "pdf", "svg"]
DPI = 300
FIGURE_SIZE = (10.0, 8.0)      # Width and height in inches
FIGURE_TITLE = None  # Use None for no title
EQUAL_ASPECT = True
ANNOTATION_DECIMALS = 1

# Display the figures in a window after saving them.
SHOW_FIGURES = True


@dataclass
class TransferCase:
    key: str
    display_name: str
    category: str
    trajectory_path: Path
    manoeuvre_path: Path | None
    trajectory: pd.DataFrame
    manoeuvres: pd.DataFrame

    @property
    def total_dv_mps(self) -> float | None:
        if self.manoeuvres.empty:
            return None
        values = pd.to_numeric(self.manoeuvres["dv_mag_mps"], errors="coerce")
        if values.notna().sum() == 0:
            return None
        return float(values.sum())


# ---------------------------------------------------------------------------
# CSV discovery and pairing
# ---------------------------------------------------------------------------


def read_csv_flexible(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read comma-, tab-, or whitespace-delimited CSV-like files."""
    try:
        return pd.read_csv(path, sep=None, engine="python", nrows=nrows)
    except Exception as first_error:
        try:
            return pd.read_csv(path, delim_whitespace=True, nrows=nrows)
        except Exception as second_error:
            raise RuntimeError(
                f"Could not read {path}. Automatic delimiter detection failed: "
                f"{first_error}; whitespace fallback failed: {second_error}"
            ) from second_error


def classify_csv(path: Path) -> str | None:
    """Return 'trajectory', 'manoeuvre', or None based on the header."""
    try:
        columns = {str(c).strip() for c in read_csv_flexible(path, nrows=0).columns}
    except Exception as exc:
        warnings.warn(f"Skipping unreadable CSV {path}: {exc}")
        return None

    if TRAJECTORY_REQUIRED.issubset(columns):
        return "trajectory"
    if MANOEUVRE_REQUIRED.issubset(columns):
        return "manoeuvre"
    return None


def category_from_name(path: Path) -> str:
    name = path.stem.lower()
    if "best_dv" in name or "best-dv" in name or "bestdv" in name:
        return "best_dv"
    if "median_dv" in name or "median-dv" in name or "mediandv" in name:
        return "median_dv"
    return "other"


def normalized_case_key(path: Path, role: str) -> str:
    """Remove role-specific filename words while retaining case identifiers."""
    stem = path.stem.lower()
    stem = re.sub(r"best[-_ ]?dv", "best_dv", stem)
    stem = re.sub(r"median[-_ ]?dv", "median_dv", stem)
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", stem) if tok]

    removable = set(GENERIC_WORDS)
    removable.update(TRAJECTORY_ROLE_WORDS if role == "trajectory" else MANOEUVRE_ROLE_WORDS)

    cleaned: list[str] = []
    i = 0
    while i < len(tokens):
        # Recombine best + dv and median + dv after tokenization.
        if i + 1 < len(tokens) and tokens[i] in {"best", "median"} and tokens[i + 1] == "dv":
            cleaned.append(f"{tokens[i]}_dv")
            i += 2
            continue
        if tokens[i] not in removable:
            cleaned.append(tokens[i])
        i += 1

    return "_".join(cleaned) or path.stem.lower()


def discover_csvs(directory: Path, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*.csv") if recursive else directory.glob("*.csv")
    return sorted(path for path in iterator if path.is_file())


def filename_matches_filters(path: Path, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    name = path.name.lower()
    normalized = name.replace("-", "_")
    return any(token.lower().replace("-", "_") in normalized for token in filters)


def pair_files(
    trajectory_paths: Sequence[Path],
    manoeuvre_paths: Sequence[Path],
) -> tuple[list[tuple[str, Path, Path | None]], list[Path]]:
    """Pair trajectory and manoeuvre files using normalized filename keys.

    Exact normalized-key matches are preferred. Remaining files are paired by
    filename similarity within the same best/median/other category.
    """
    trajectory_info = [
        (normalized_case_key(path, "trajectory"), category_from_name(path), path)
        for path in trajectory_paths
    ]
    manoeuvre_info = [
        (normalized_case_key(path, "manoeuvre"), category_from_name(path), path)
        for path in manoeuvre_paths
    ]

    unused_manoeuvres = set(range(len(manoeuvre_info)))
    pairs: list[tuple[str, Path, Path | None]] = []

    for traj_key, traj_category, traj_path in trajectory_info:
        exact = [
            idx
            for idx in unused_manoeuvres
            if manoeuvre_info[idx][0] == traj_key
        ]

        chosen: int | None = None
        if len(exact) == 1:
            chosen = exact[0]
        elif len(exact) > 1:
            same_category = [
                idx for idx in exact if manoeuvre_info[idx][1] == traj_category
            ]
            chosen = same_category[0] if same_category else exact[0]
            warnings.warn(
                f"Multiple manoeuvre files exactly match {traj_path.name}; "
                f"using {manoeuvre_info[chosen][2].name}."
            )
        else:
            candidates = [
                idx
                for idx in unused_manoeuvres
                if manoeuvre_info[idx][1] == traj_category
            ]
            if not candidates:
                candidates = list(unused_manoeuvres)

            # Prefer a prefix relationship before fuzzy matching. This handles
            # filenames such as:
            #   best_dv_684234_maneuvers.csv
            #   best_dv_684234_TCM-1_plus_TCM-2_plus_TCM-3_plus_ACQ_trajectory.csv
            # where the manoeuvre key is the identifying prefix of the longer
            # trajectory key.
            prefix_matches = [
                idx
                for idx in candidates
                if (
                    traj_key.startswith(manoeuvre_info[idx][0] + "_")
                    or manoeuvre_info[idx][0].startswith(traj_key + "_")
                )
            ]

            if len(prefix_matches) == 1:
                chosen = prefix_matches[0]
            elif len(prefix_matches) > 1:
                # Use the longest matching key because it is the most specific.
                chosen = max(
                    prefix_matches,
                    key=lambda idx: len(manoeuvre_info[idx][0]),
                )
                warnings.warn(
                    f"Multiple manoeuvre files partially match {traj_path.name}; "
                    f"using {manoeuvre_info[chosen][2].name}."
                )
            else:
                scored = [
                    (
                        SequenceMatcher(None, traj_key, manoeuvre_info[idx][0]).ratio(),
                        idx,
                    )
                    for idx in candidates
                ]
                if scored:
                    score, best_idx = max(scored)
                    if score >= 0.45:
                        chosen = best_idx

        if chosen is None:
            pairs.append((traj_key, traj_path, None))
        else:
            unused_manoeuvres.remove(chosen)
            pairs.append((traj_key, traj_path, manoeuvre_info[chosen][2]))

    unmatched = [manoeuvre_info[idx][2] for idx in sorted(unused_manoeuvres)]
    return pairs, unmatched


def clean_trajectory(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = TRAJECTORY_REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing trajectory columns: {sorted(missing)}")

    out = df.copy()
    numeric_columns = [
        "jdtdb",
        "t_days_since_departure",
        "x_geo_j2000_km",
        "y_geo_j2000_km",
        "z_geo_j2000_km",
    ]
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(
        subset=["jdtdb", "x_geo_j2000_km", "y_geo_j2000_km", "z_geo_j2000_km"]
    )
    sort_column = "t_days_since_departure" if "t_days_since_departure" in out.columns else "jdtdb"
    out = out.sort_values(sort_column).reset_index(drop=True)

    if len(out) < 2:
        raise ValueError(f"{path} contains fewer than two valid trajectory rows.")
    return out


def clean_manoeuvres(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = MANOEUVRE_REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing manoeuvre columns: {sorted(missing)}")

    out = df.copy()
    out["fraction"] = pd.to_numeric(out["fraction"], errors="coerce")
    out["dv_mag_mps"] = pd.to_numeric(out["dv_mag_mps"], errors="coerce")
    out = out.dropna(subset=["fraction", "dv_mag_mps"]).copy()

    outside = ~out["fraction"].between(0.0, 1.0)
    if outside.any():
        warnings.warn(
            f"Clipping {int(outside.sum())} manoeuvre fractions outside [0, 1] in {path.name}."
        )
        out["fraction"] = out["fraction"].clip(0.0, 1.0)

    return out.sort_values("fraction").reset_index(drop=True)


def prettify_case_name(key: str, category: str) -> str:
    base = key
    base = base.replace("best_dv", "").replace("median_dv", "")
    base = re.sub(r"_+", " ", base).strip().title()

    category_text = {
        "best_dv": "Best $\\Delta v$",
        "median_dv": "Median $\\Delta v$",
        "other": "Trajectory",
    }[category]
    return f"{base} ({category_text})" if base else category_text


def load_cases(
    trajectory_dir: Path,
    manoeuvre_dir: Path,
    recursive: bool,
    name_filters: Sequence[str],
) -> list[TransferCase]:
    trajectory_candidates = discover_csvs(trajectory_dir, recursive)
    manoeuvre_candidates = (
        trajectory_candidates
        if manoeuvre_dir.resolve() == trajectory_dir.resolve()
        else discover_csvs(manoeuvre_dir, recursive)
    )

    trajectory_paths = [
        path
        for path in trajectory_candidates
        if filename_matches_filters(path, name_filters) and classify_csv(path) == "trajectory"
    ]
    manoeuvre_paths = [
        path
        for path in manoeuvre_candidates
        if filename_matches_filters(path, name_filters) and classify_csv(path) == "manoeuvre"
    ]

    if not trajectory_paths:
        filters_text = ", ".join(name_filters) if name_filters else "none"
        raise FileNotFoundError(
            f"No trajectory CSVs were found in {trajectory_dir} using filename filters: {filters_text}."
        )

    pairs, unmatched_manoeuvres = pair_files(trajectory_paths, manoeuvre_paths)
    for path in unmatched_manoeuvres:
        warnings.warn(f"No trajectory file could be paired with manoeuvre file {path.name}.")

    cases: list[TransferCase] = []
    for key, trajectory_path, manoeuvre_path in pairs:
        trajectory = clean_trajectory(read_csv_flexible(trajectory_path), trajectory_path)
        if manoeuvre_path is None:
            warnings.warn(
                f"No manoeuvre CSV was paired with {trajectory_path.name}; "
                "the trajectory will be plotted with total Δv shown as n/a."
            )
            manoeuvres = pd.DataFrame(columns=["fraction", "dv_mag_mps", "name", "label"])
        else:
            manoeuvres = clean_manoeuvres(read_csv_flexible(manoeuvre_path), manoeuvre_path)

        category = category_from_name(trajectory_path)
        cases.append(
            TransferCase(
                key=key,
                display_name=prettify_case_name(key, category),
                category=category,
                trajectory_path=trajectory_path,
                manoeuvre_path=manoeuvre_path,
                trajectory=trajectory,
                manoeuvres=manoeuvres,
            )
        )

    # Stable order: best, median, then other, followed by name.
    category_order = {"best_dv": 0, "median_dv": 1, "other": 2}
    return sorted(cases, key=lambda case: (category_order[case.category], case.display_name))


# ---------------------------------------------------------------------------
# Trajectory interpolation and SPICE Moon ephemeris
# ---------------------------------------------------------------------------


def normalized_transfer_coordinate(trajectory: pd.DataFrame) -> np.ndarray:
    if (
        "t_days_since_departure" in trajectory.columns
        and trajectory["t_days_since_departure"].notna().sum() >= 2
    ):
        values = trajectory["t_days_since_departure"].to_numpy(dtype=float)
    else:
        values = trajectory["jdtdb"].to_numpy(dtype=float)

    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return np.linspace(0.0, 1.0, len(trajectory))
    return (values - minimum) / (maximum - minimum)


def manoeuvre_position(trajectory: pd.DataFrame, fraction: float) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    tau = normalized_transfer_coordinate(trajectory)
    xyz = trajectory[
        ["x_geo_j2000_km", "y_geo_j2000_km", "z_geo_j2000_km"]
    ].to_numpy(dtype=float)

    # Remove duplicate normalized times to keep interpolation deterministic.
    tau_unique, unique_indices = np.unique(tau, return_index=True)
    xyz_unique = xyz[unique_indices]
    if len(tau_unique) == 1:
        return xyz_unique[0]

    return np.array(
        [np.interp(fraction, tau_unique, xyz_unique[:, axis]) for axis in range(3)]
    )


def load_spice_moon_path(
    kernels: Sequence[Path],
    jdtdb_min: float,
    jdtdb_max: float,
    samples: int,
) -> np.ndarray | None:
    if not kernels:
        warnings.warn(
            "No SPICE kernel was supplied. Earth will be plotted, but the Moon will be omitted. "
            "Add an SPK or meta-kernel to SPICE_KERNELS in the USER CONFIGURATION section."
        )
        return None

    try:
        import spiceypy as spice
    except ImportError:
        warnings.warn(
            "spiceypy is not installed. Earth will be plotted, but the Moon will be omitted. "
            "Install it with: pip install spiceypy"
        )
        return None

    for kernel in kernels:
        if not kernel.exists():
            raise FileNotFoundError(f"SPICE kernel does not exist: {kernel}")

    spice.kclear()
    try:
        for kernel in kernels:
            spice.furnsh(str(kernel))

        jdtdb = np.linspace(jdtdb_min, jdtdb_max, max(2, int(samples)))
        et = (jdtdb - J2000_JD_TDB) * SECONDS_PER_DAY
        positions = []
        for epoch in et:
            position, _ = spice.spkpos("MOON", float(epoch), "J2000", "NONE", "EARTH")
            positions.append(position)
        return np.asarray(positions, dtype=float)
    except Exception as exc:
        warnings.warn(f"SPICE could not generate the Moon ephemeris: {exc}")
        return None
    finally:
        spice.kclear()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def coordinate_indices(view: str) -> tuple[int, int] | None:
    return {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }.get(view)


def axis_label(axis_index: int) -> str:
    return ["x (km)", "y (km)", "z (km)"][axis_index]


def manoeuvre_text(row: pd.Series, number: int, decimals: int) -> str:
    # Use only the manoeuvre ``name`` column for annotations.
    name = ""
    if "name" in row.index and pd.notna(row["name"]):
        candidate = str(row["name"]).strip()
        if candidate and candidate.lower() != "nan":
            name = candidate
    if not name:
        name = f"M{number}"
    return f"{name}: {float(row['dv_mag_mps']):.{decimals}f} m/s"


def collect_plot_points(cases: Sequence[TransferCase], moon_xyz: np.ndarray | None) -> np.ndarray:
    arrays = [
        case.trajectory[
            ["x_geo_j2000_km", "y_geo_j2000_km", "z_geo_j2000_km"]
        ].to_numpy(dtype=float)
        for case in cases
    ]
    arrays.append(np.zeros((1, 3)))
    if moon_xyz is not None and moon_xyz.size:
        arrays.append(moon_xyz)
    return np.vstack(arrays)


def set_equal_3d_limits(ax, points: np.ndarray) -> None:
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    centres = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.nanmax(maxs - mins))
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0
    ax.set_xlim(centres[0] - radius, centres[0] + radius)
    ax.set_ylim(centres[1] - radius, centres[1] + radius)
    ax.set_zlim(centres[2] - radius, centres[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_view(
    cases: Sequence[TransferCase],
    moon_xyz: np.ndarray | None,
    view: str,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    title: str | None,
    equal_aspect: bool,
    annotation_decimals: int,
    figure_size: tuple[float, float],
    show: bool,
) -> list[Path]:
    is_3d = view == "3d"
    if is_3d:
        fig = plt.figure(figsize=figure_size, constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)

    cmap = plt.get_cmap("tab20")
    legend_handles: list[Line2D] = []
    all_points = collect_plot_points(cases, moon_xyz)
    spans = np.ptp(all_points, axis=0)
    spans[spans <= 0.0] = 1.0

    # Earth at the geocentric origin.
    if is_3d:
        ax.scatter(0.0, 0.0, 0.0, s=100, marker="o", c="royalblue", edgecolors="black", zorder=8)
        ax.text(0.0, 0.0, 0.0, "  Earth", fontsize=9)
    else:
        axes = coordinate_indices(view)
        assert axes is not None
        ax.scatter(0.0, 0.0, s=100, marker="o", c="royalblue", edgecolors="black", zorder=8)
        ax.annotate("Earth", (0.0, 0.0), xytext=(7, 7), textcoords="offset points", fontsize=9)

    earth_handle = Line2D(
        [0], [0], marker="o", linestyle="None", markerfacecolor="royalblue",
        markeredgecolor="black", markersize=8, label="Earth"
    )

    moon_handle: Line2D | None = None
    if moon_xyz is not None and len(moon_xyz) >= 2:
        if is_3d:
            ax.plot(
                moon_xyz[:, 0], moon_xyz[:, 1], moon_xyz[:, 2],
                color="0.45", linewidth=1.1, alpha=0.8, zorder=1
            )
            ax.scatter(*moon_xyz[0], marker="o", s=35, c="0.65", edgecolors="black", zorder=7)
            ax.scatter(*moon_xyz[-1], marker="x", s=45, c="0.25", zorder=7)
        else:
            i, j = coordinate_indices(view)  # type: ignore[misc]
            ax.plot(
                moon_xyz[:, i], moon_xyz[:, j],
                color="0.45", linewidth=1.1, alpha=0.8, zorder=1
            )
            ax.scatter(
                moon_xyz[0, i], moon_xyz[0, j], marker="o", s=35,
                c="0.65", edgecolors="black", zorder=7
            )
            ax.scatter(
                moon_xyz[-1, i], moon_xyz[-1, j], marker="x", s=45,
                c="0.25", zorder=7
            )
        moon_handle = Line2D(
            [0], [0], color="0.45", linewidth=1.1, marker="o",
            markerfacecolor="0.65", markeredgecolor="black", markersize=5,
            label="Moon"
        )

    for case_index, case in enumerate(cases):
        color = cmap(case_index % cmap.N)
        marker = MARKERS[case_index % len(MARKERS)]
        linestyle = LINESTYLES.get(case.category, LINESTYLES["other"])
        xyz = case.trajectory[
            ["x_geo_j2000_km", "y_geo_j2000_km", "z_geo_j2000_km"]
        ].to_numpy(dtype=float)

        if is_3d:
            ax.plot(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                color=color, linestyle=linestyle, linewidth=1.8, zorder=3
            )
            ax.scatter(*xyz[0], marker="o", s=55, facecolors="none", edgecolors=[color], linewidths=1.6, zorder=9)
            ax.scatter(*xyz[-1], marker="x", s=60, c=[color], linewidths=1.8, zorder=9)
        else:
            i, j = coordinate_indices(view)  # type: ignore[misc]
            ax.plot(
                xyz[:, i], xyz[:, j],
                color=color, linestyle=linestyle, linewidth=1.8, zorder=3
            )
            ax.scatter(
                xyz[0, i], xyz[0, j], marker="o", s=55,
                facecolors="none", edgecolors=[color], linewidths=1.6, zorder=9
            )
            ax.scatter(
                xyz[-1, i], xyz[-1, j], marker="x", s=60,
                c=[color], linewidths=1.8, zorder=9
            )

        for manoeuvre_index, (_, row) in enumerate(case.manoeuvres.iterrows(), start=1):
            position = manoeuvre_position(case.trajectory, float(row["fraction"]))
            label = manoeuvre_text(row, manoeuvre_index, annotation_decimals)

            if is_3d:
                ax.scatter(
                    position[0], position[1], position[2],
                    marker=marker, s=70, c=[color], edgecolors="black",
                    linewidths=0.6, zorder=10
                )
                direction = -1.0 if manoeuvre_index % 2 == 0 else 1.0
                offset = direction * np.array([0.012, 0.016, 0.014]) * spans
                text_position = position + offset
                ax.text(
                    text_position[0], text_position[1], text_position[2], label,
                    color=color, fontsize=8,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.85},
                    zorder=11,
                )
            else:
                i, j = coordinate_indices(view)  # type: ignore[misc]
                ax.scatter(
                    position[i], position[j], marker=marker, s=70,
                    c=[color], edgecolors="black", linewidths=0.6, zorder=10
                )
                offsets = [(8, 10), (8, -17), (-8, 10), (-8, -17)]
                xytext = offsets[(manoeuvre_index - 1) % len(offsets)]
                ax.annotate(
                    label,
                    xy=(position[i], position[j]),
                    xytext=xytext,
                    textcoords="offset points",
                    ha="left" if xytext[0] >= 0 else "right",
                    va="bottom" if xytext[1] >= 0 else "top",
                    fontsize=8,
                    color=color,
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7},
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.85},
                    zorder=11,
                )

        total_dv = case.total_dv_mps
        dv_text = "n/a" if total_dv is None else f"{total_dv:.1f} m/s"
        legend_prefix = {
            "best_dv": "Best $\Delta V$",
            "median_dv": "Median $\Delta V$",
            "other": "Delta v",
        }[case.category]
        legend_handles.append(
            Line2D(
                [0], [0], color=color, linestyle=linestyle, linewidth=1.8,
                marker=marker, markerfacecolor=color, markeredgecolor="black",
                markersize=7, label=f"{legend_prefix} = {dv_text}"
            )
        )

    start_handle = Line2D(
        [0], [0], marker="o", linestyle="None", markerfacecolor="none",
        markeredgecolor="black", markersize=7, label="Transfer start"
    )
    end_handle = Line2D(
        [0], [0], marker="x", linestyle="None", color="black",
        markersize=7, label="Transfer end"
    )

    if is_3d:
        ax.set_xlabel(axis_label(0), labelpad=10)
        ax.set_ylabel(axis_label(1), labelpad=10)
        ax.set_zlabel(axis_label(2), labelpad=10)
        if equal_aspect:
            set_equal_3d_limits(ax, all_points)
        ax.view_init(elev=25, azim=-55)
    else:
        i, j = coordinate_indices(view)  # type: ignore[misc]
        ax.set_xlabel(axis_label(i))
        ax.set_ylabel(axis_label(j))
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.5, alpha=0.35)

    ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 4), useMathText=True)
    if is_3d:
        # Matplotlib's 3-D axes do not always honour ticklabel_format for z.
        ax.zaxis.get_major_formatter().set_powerlimits((-3, 4))

    if title:
        ax.set_title(title)

    handles = legend_handles + [start_handle, end_handle, earth_handle]
    if moon_handle is not None:
        handles.append(moon_handle)
    ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.95)

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        extension = extension.lower().lstrip(".")
        output_path = output_dir / f"transfer_trajectories_{view}.{extension}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written.append(output_path)

    if show:
        plt.show()
    return written


# ---------------------------------------------------------------------------
# PyCharm entry point
# ---------------------------------------------------------------------------


def validate_configuration() -> None:
    valid_views = {"3d", "xy", "xz", "yz"}
    invalid_views = [view for view in VIEWS if view not in valid_views]
    if invalid_views:
        raise ValueError(
            f"Invalid VIEWS entries: {invalid_views}. "
            f"Allowed values are: {sorted(valid_views)}."
        )

    if not VIEWS:
        raise ValueError("VIEWS must contain at least one plot view.")
    if not OUTPUT_FORMATS:
        raise ValueError("OUTPUT_FORMATS must contain at least one file format.")
    if MOON_SAMPLES < 2:
        raise ValueError("MOON_SAMPLES must be at least 2.")
    if DPI <= 0:
        raise ValueError("DPI must be positive.")
    if len(FIGURE_SIZE) != 2 or min(FIGURE_SIZE) <= 0:
        raise ValueError("FIGURE_SIZE must contain two positive values.")

    if not TRAJECTORY_DIR.is_dir():
        raise NotADirectoryError(
            "TRAJECTORY_DIR does not exist. Edit the USER CONFIGURATION section: "
            f"{TRAJECTORY_DIR}"
        )
    if not MANOEUVRE_DIR.is_dir():
        raise NotADirectoryError(
            "MANOEUVRE_DIR does not exist. Edit the USER CONFIGURATION section: "
            f"{MANOEUVRE_DIR}"
        )


def main() -> int:
    validate_configuration()

    cases = load_cases(
        trajectory_dir=TRAJECTORY_DIR,
        manoeuvre_dir=MANOEUVRE_DIR,
        recursive=RECURSIVE_SEARCH,
        name_filters=NAME_FILTERS,
    )

    jdtdb_min = min(float(case.trajectory["jdtdb"].min()) for case in cases)
    jdtdb_max = max(float(case.trajectory["jdtdb"].max()) for case in cases)
    moon_xyz = load_spice_moon_path(
        kernels=SPICE_KERNELS,
        jdtdb_min=jdtdb_min,
        jdtdb_max=jdtdb_max,
        samples=MOON_SAMPLES,
    )

    print(f"Loaded {len(cases)} trajectory case(s):")
    for case in cases:
        manoeuvre_name = case.manoeuvre_path.name if case.manoeuvre_path else "not paired"
        total = case.total_dv_mps
        total_text = "n/a" if total is None else f"{total:.3f} m/s"
        print(
            f"  - {case.trajectory_path.name} | manoeuvres: {manoeuvre_name} | "
            f"total Delta-v: {total_text}"
        )

    outputs: list[Path] = []
    for view in VIEWS:
        outputs.extend(
            plot_view(
                cases=cases,
                moon_xyz=moon_xyz,
                view=view,
                output_dir=OUTPUT_DIR,
                formats=OUTPUT_FORMATS,
                dpi=DPI,
                title=FIGURE_TITLE,
                equal_aspect=EQUAL_ASPECT,
                annotation_decimals=ANNOTATION_DECIMALS,
                figure_size=FIGURE_SIZE,
                show=False,
            )
        )

    print("Written figures:")
    for output in outputs:
        print(f"  {output.resolve()}")

    # Calling plt.show() once keeps every requested view open together.
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")

    return 0


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        raise
