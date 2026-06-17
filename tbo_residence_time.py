from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Circle, Ellipse


# ============================================================
# Constants
# ============================================================

AU_KM = 149_597_870.7

ONE_EH_AU = 0.01

HALO_Y_RADIUS_KM = 800_000.0
HALO_Z_RADIUS_KM = 500_000.0

HALO_Y_RADIUS_AU = HALO_Y_RADIUS_KM / AU_KM
HALO_Z_RADIUS_AU = HALO_Z_RADIUS_KM / AU_KM


# ============================================================
# Utilities
# ============================================================

def normalize_object_id(x) -> str:
    return str(x).strip()


def infer_population_from_object_id(object_id: str) -> str:
    """
    Synthetic objects start with NESC.
    Everything else is treated as real.
    """
    object_id = normalize_object_id(object_id)
    return "synthetic" if object_id.upper().startswith("NESC") else "real"


def read_space_separated_csv(path: Path, **kwargs):
    """
    Read files written with sep=' '.

    Intended for files written by:
        df.to_csv(path, sep=' ', index=False)
    """
    return pd.read_csv(
        path,
        sep=" ",
        engine="python",
        skipinitialspace=True,
        **kwargs,
    )


def get_file_columns(file: Path):
    header = read_space_separated_csv(file, nrows=0)
    return list(header.columns)


def filter_candidate_files(
    trajectory_folder: Path,
    trajectory_file_glob: str,
    exclude_file_keywords: list[str] | None = None,
) -> list[Path]:
    files = sorted(Path(trajectory_folder).glob(trajectory_file_glob))

    if exclude_file_keywords is not None:
        files = [
            f for f in files
            if not any(keyword in f.name for keyword in exclude_file_keywords)
        ]

    return files


# ============================================================
# Residence-time accumulation
# ============================================================

def accumulate_synthetic_residence_time_maps(
    trajectory_folder: Path,
    object_id_col: str,
    epoch_col: str,
    x_col: str,
    y_col: str,
    z_col: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    trajectory_file_glob: str = "*.csv",
    progress_every: int = 100,
    overlay_real_tbos: bool = True,
    real_overlay_max_points_per_object: int = 2_000,
    exclude_file_keywords: list[str] | None = None,
):
    """
    Build synthetic-only residence-time histograms in:
        - XY projection
        - YZ projection

    Color quantity:
        total synthetic residence time in days.

    Optionally collect real TBO trajectories for overlay.
    """

    trajectory_folder = Path(trajectory_folder)

    files = filter_candidate_files(
        trajectory_folder=trajectory_folder,
        trajectory_file_glob=trajectory_file_glob,
        exclude_file_keywords=exclude_file_keywords,
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No candidate files matching {trajectory_file_glob} found in {trajectory_folder}"
        )

    H_xy = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=float)
    H_yz = np.zeros((len(y_edges) - 1, len(z_edges) - 1), dtype=float)

    real_overlay = []

    n_files_total = len(files)
    n_objects_processed = 0
    n_synthetic_used = 0
    n_real_overlay = 0
    total_synthetic_time_days = 0.0

    t0 = time.time()

    print(f"Found {n_files_total:,} candidate files")
    print("Accumulating synthetic residence-time distribution...")

    for i, file in enumerate(files, start=1):
        try:
            columns = get_file_columns(file)
        except Exception as e:
            print(f"Skipping {file.name}: could not read header. Error: {e}")
            continue

        has_object_id_col = object_id_col in columns

        required_cols = [epoch_col, x_col, y_col, z_col]
        missing = [c for c in required_cols if c not in columns]

        if missing:
            print(f"Skipping {file.name}: missing columns {missing}")
            continue

        usecols = [epoch_col, x_col, y_col, z_col]
        if has_object_id_col:
            usecols.append(object_id_col)

        try:
            df = read_space_separated_csv(file, usecols=usecols)
        except Exception as e:
            print(f"Skipping {file.name}: could not read data. Error: {e}")
            continue

        if has_object_id_col:
            object_ids = df[object_id_col].dropna().astype(str).unique()
            object_id = normalize_object_id(object_ids[0]) if len(object_ids) > 0 else file.stem
        else:
            object_id = normalize_object_id(file.stem)
            df[object_id_col] = object_id

        population = infer_population_from_object_id(object_id)

        df[epoch_col] = pd.to_numeric(df[epoch_col], errors="coerce")
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df[z_col] = pd.to_numeric(df[z_col], errors="coerce")

        df = df.dropna(subset=[epoch_col, x_col, y_col, z_col])

        if len(df) < 2:
            n_objects_processed += 1
            continue

        df = df.sort_values(epoch_col)

        t = df[epoch_col].to_numpy(dtype=float)
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        z = df[z_col].to_numpy(dtype=float)

        if population == "real":
            if overlay_real_tbos:
                x_real, y_real, z_real = downsample_real_trajectory(
                    x=x,
                    y=y,
                    z=z,
                    max_points=real_overlay_max_points_per_object,
                )

                real_overlay.append(
                    {
                        "object_id": object_id,
                        "x": x_real,
                        "y": y_real,
                        "z": z_real,
                    }
                )
                n_real_overlay += 1

            n_objects_processed += 1
            continue

        # Synthetic only for residence-time map
        dt = np.diff(t)

        valid = dt > 0
        if not np.any(valid):
            n_objects_processed += 1
            continue

        dt = dt[valid]
        x_mid = 0.5 * (x[:-1] + x[1:])[valid]
        y_mid = 0.5 * (y[:-1] + y[1:])[valid]
        z_mid = 0.5 * (z[:-1] + z[1:])[valid]

        H_xy += np.histogram2d(
            x_mid,
            y_mid,
            bins=[x_edges, y_edges],
            weights=dt,
        )[0]

        H_yz += np.histogram2d(
            y_mid,
            z_mid,
            bins=[y_edges, z_edges],
            weights=dt,
        )[0]

        total_synthetic_time_days += np.sum(dt)

        n_objects_processed += 1
        n_synthetic_used += 1

        if i == 1 or i == n_files_total or i % progress_every == 0:
            elapsed = time.time() - t0
            pct = 100.0 * i / n_files_total
            files_per_sec = i / elapsed if elapsed > 0 else np.nan
            eta_sec = (n_files_total - i) / files_per_sec if files_per_sec > 0 else np.nan

            print(
                f"[{i:,}/{n_files_total:,}] {pct:6.2f}% | "
                f"current: {file.name} | "
                f"synthetic used: {n_synthetic_used:,} | "
                f"real overlay: {n_real_overlay:,} | "
                f"synthetic residence time: {total_synthetic_time_days:,.1f} d | "
                f"elapsed: {elapsed / 60:.1f} min | "
                f"ETA: {eta_sec / 60:.1f} min"
            )

    return {
        "H_xy": H_xy,
        "H_yz": H_yz,
        "real_overlay": real_overlay,
        "n_objects_processed": n_objects_processed,
        "n_synthetic_used": n_synthetic_used,
        "n_real_overlay": n_real_overlay,
        "total_synthetic_time_days": total_synthetic_time_days,
    }


# ============================================================
# Real TBO overlay loading
# ============================================================

def downsample_real_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        return x[idx], y[idx], z[idx]

    return x, y, z


def load_real_tbo_overlays(
    trajectory_folder: Path,
    object_id_col: str,
    epoch_col: str,
    x_col: str,
    y_col: str,
    z_col: str,
    trajectory_file_glob: str = "*.csv",
    exclude_file_keywords: list[str] | None = None,
    real_overlay_max_points_per_object: int = 3_000,
    progress_every: int = 100,
) -> list[dict]:
    """
    Re-read only real TBO trajectories for overlay.

    This is useful in RUN_MODE = 'plot_from_grid', where the synthetic residence
    grids are loaded from CSV and trajectories are not reprocessed.
    """

    trajectory_folder = Path(trajectory_folder)

    files = filter_candidate_files(
        trajectory_folder=trajectory_folder,
        trajectory_file_glob=trajectory_file_glob,
        exclude_file_keywords=exclude_file_keywords,
    )

    real_overlay = []

    print("Loading real TBO overlays...")

    for i, file in enumerate(files, start=1):
        try:
            columns = get_file_columns(file)
        except Exception:
            continue

        has_object_id_col = object_id_col in columns

        required_cols = [epoch_col, x_col, y_col, z_col]
        missing = [c for c in required_cols if c not in columns]
        if missing:
            continue

        usecols = [epoch_col, x_col, y_col, z_col]
        if has_object_id_col:
            usecols.append(object_id_col)

        try:
            df = read_space_separated_csv(file, usecols=usecols)
        except Exception:
            continue

        if has_object_id_col:
            object_ids = df[object_id_col].dropna().astype(str).unique()
            object_id = normalize_object_id(object_ids[0]) if len(object_ids) > 0 else file.stem
        else:
            object_id = normalize_object_id(file.stem)

        population = infer_population_from_object_id(object_id)

        if population != "real":
            continue

        df[epoch_col] = pd.to_numeric(df[epoch_col], errors="coerce")
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df[z_col] = pd.to_numeric(df[z_col], errors="coerce")

        df = df.dropna(subset=[epoch_col, x_col, y_col, z_col])

        if len(df) < 2:
            continue

        df = df.sort_values(epoch_col)

        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        z = df[z_col].to_numpy(dtype=float)

        x_real, y_real, z_real = downsample_real_trajectory(
            x=x,
            y=y,
            z=z,
            max_points=real_overlay_max_points_per_object,
        )

        real_overlay.append(
            {
                "object_id": object_id,
                "x": x_real,
                "y": y_real,
                "z": z_real,
            }
        )

        if i == 1 or i == len(files) or i % progress_every == 0:
            print(
                f"[real overlay scan {i:,}/{len(files):,}] "
                f"loaded real TBOs: {len(real_overlay):,}"
            )

    print(f"Loaded {len(real_overlay):,} real TBO overlays")

    return real_overlay


# ============================================================
# Grid CSV saving/loading
# ============================================================

def save_histogram_csv(
    H: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_name: str,
    y_name: str,
    output_csv: Path,
):
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    rows = []

    for i, xc in enumerate(x_centers):
        for j, yc in enumerate(y_centers):
            rows.append(
                {
                    x_name: xc,
                    y_name: yc,
                    "residence_time_days": H[i, j],
                }
            )

    pd.DataFrame(rows).to_csv(output_csv, index=False)


def load_histogram_csv(
    grid_csv: Path,
    x_name: str,
    y_name: str,
    value_name: str = "residence_time_days",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a saved residence-time grid CSV and reconstruct:

        H
        x_edges
        y_edges

    Assumes the CSV was written by save_histogram_csv and contains bin centers.
    """

    df = pd.read_csv(grid_csv)

    required_cols = [x_name, y_name, value_name]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Grid CSV is missing columns {missing}. "
            f"Columns found were: {list(df.columns)}"
        )

    x_centers = np.sort(df[x_name].unique())
    y_centers = np.sort(df[y_name].unique())

    nx = len(x_centers)
    ny = len(y_centers)

    if nx > 1:
        dx = np.median(np.diff(x_centers))
    else:
        dx = 1.0

    if ny > 1:
        dy = np.median(np.diff(y_centers))
    else:
        dy = 1.0

    x_edges = np.concatenate(
        [
            [x_centers[0] - 0.5 * dx],
            0.5 * (x_centers[:-1] + x_centers[1:]),
            [x_centers[-1] + 0.5 * dx],
        ]
    )

    y_edges = np.concatenate(
        [
            [y_centers[0] - 0.5 * dy],
            0.5 * (y_centers[:-1] + y_centers[1:]),
            [y_centers[-1] + 0.5 * dy],
        ]
    )

    pivot = df.pivot(
        index=x_name,
        columns=y_name,
        values=value_name,
    )

    pivot = pivot.reindex(index=x_centers, columns=y_centers)

    H = pivot.to_numpy(dtype=float)

    return H, x_edges, y_edges


# ============================================================
# Plot overlays
# ============================================================

def add_common_xy_overlays(
    ax,
    overlay_1eh: bool = True,
    overlay_halo: bool = True,
):
    """
    XY projection overlays.

    Earth is at origin.
    1 EH is a circle centered at origin.
    The representative halo orbit is shown as its Y-extent at x = 0.
    """

    ax.scatter(
        [0.0],
        [0.0],
        marker="o",
        s=55,
        c="deepskyblue",
        edgecolors="black",
        linewidths=0.8,
        zorder=8,
        label="Earth",
    )

    if overlay_1eh:
        circle = Circle(
            (0.0, 0.0),
            ONE_EH_AU,
            fill=False,
            color="cyan",
            linestyle="--",
            linewidth=1.5,
            zorder=7,
            label="1 EH = 0.01 AU",
        )
        ax.add_patch(circle)

    if overlay_halo:
        ax.plot(
            [0.01, 0.01],
            [-HALO_Y_RADIUS_AU, HALO_Y_RADIUS_AU],
            color="magenta",
            linestyle="-",
            linewidth=2.0,
            zorder=7,
            label="Representative halo projection",
        )


def add_common_yz_overlays(
    ax,
    overlay_1eh: bool = True,
    overlay_halo: bool = True,
):
    """
    YZ projection overlays.

    Earth is at origin.
    1 EH is a circle centered at origin.
    The representative halo orbit is an ellipse in YZ.
    """

    ax.scatter(
        [0.0],
        [0.0],
        marker="o",
        s=55,
        c="deepskyblue",
        edgecolors="black",
        linewidths=0.8,
        zorder=8,
        label="Earth",
    )

    if overlay_1eh:
        circle = Circle(
            (0.0, 0.0),
            ONE_EH_AU,
            fill=False,
            color="cyan",
            linestyle="--",
            linewidth=1.5,
            zorder=7,
            label="1 EH = 0.01 AU",
        )
        ax.add_patch(circle)

    if overlay_halo:
        ellipse = Ellipse(
            (0.0, 0.0),
            width=2.0 * HALO_Y_RADIUS_AU,
            height=2.0 * HALO_Z_RADIUS_AU,
            fill=False,
            color="magenta",
            linestyle="-",
            linewidth=2.0,
            zorder=7,
            label="Representative halo orbit",
        )
        ax.add_patch(ellipse)


def overlay_real_tbos_xy(ax, real_overlay, max_labelled: int = 5):
    if len(real_overlay) == 0:
        return

    for k, obj in enumerate(real_overlay):
        label = "Real TBO trajectories" if k == 0 else None

        ax.plot(
            obj["x"],
            obj["y"],
            color="grey",
            linewidth=1.2,
            alpha=0.9,
            zorder=9,
            label=label,
        )

        # ax.scatter(
        #     obj["x"][0],
        #     obj["y"][0],
        #     s=20,
        #     color="white",
        #     edgecolors="black",
        #     linewidths=0.5,
        #     zorder=10,
        # )

        if k < max_labelled:
            ax.text(
                obj["x"][0],
                obj["y"][0],
                f" {obj['object_id']}",
                color="grey",
                fontsize=8,
                zorder=11,
            )


def overlay_real_tbos_yz(ax, real_overlay, max_labelled: int = 5):
    if len(real_overlay) == 0:
        return

    for k, obj in enumerate(real_overlay):
        label = "Real TBO trajectories" if k == 0 else None

        ax.plot(
            obj["y"],
            obj["z"],
            color="white",
            linewidth=1.2,
            alpha=0.9,
            zorder=9,
            label=label,
        )

        ax.scatter(
            obj["y"][0],
            obj["z"][0],
            s=20,
            color="white",
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )

        if k < max_labelled:
            ax.text(
                obj["y"][0],
                obj["z"][0],
                f" {obj['object_id']}",
                color="white",
                fontsize=8,
                zorder=11,
            )


# ============================================================
# Plotting
# ============================================================

def plot_residence_contour(
    H: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    projection: str,
    real_overlay: list | None = None,
    overlay_real: bool = True,
    sigma: float = 1.0,
    n_levels: int = 25,
    cmap: str = "hot_r",
    vmin: float = 0.0,
    vmax: float | None = None,
):
    """
    Plot a residence-time contour map.

    projection:
        "xy" or "yz"
    """

    Z = H.T.copy()

    if sigma is not None and sigma > 0:
        Z_plot = gaussian_filter(Z, sigma=sigma)
    else:
        Z_plot = Z

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="xy")

    if vmax is None:
        vmax = np.nanmax(Z_plot)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    levels = np.linspace(vmin, vmax, n_levels)

    fig, ax = plt.subplots(figsize=(8, 6))

    cf = ax.contourf(
        Xc,
        Yc,
        Z_plot,
        levels=levels,
        cmap=cmap,
        extend="max",
    )

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Synthetic TBO residence time (days)")

    if projection.lower() == "xy":
        add_common_xy_overlays(ax)
        if overlay_real and real_overlay is not None:
            overlay_real_tbos_xy(ax, real_overlay)

    elif projection.lower() == "yz":
        add_common_yz_overlays(ax)
        if overlay_real and real_overlay is not None:
            overlay_real_tbos_yz(ax, real_overlay)

    else:
        raise ValueError("projection must be 'xy' or 'yz'")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)

    ax.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Run mode
    # ------------------------------------------------------------

    # Options:
    #   "accumulate_and_plot"
    #       Process all trajectory files, save grid CSVs, and make figures.
    #
    #   "plot_from_grid"
    #       Skip trajectory processing and regenerate figures from the saved
    #       grid CSVs. Real overlays can still be loaded separately.
    RUN_MODE = "plot_from_grid"

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------

    TRAJECTORY_FOLDER = Path(
        r"/media/aeromec/Seagate Desktop Drive/minimoon_files_oorb"
    )

    OUTPUT_DIR = Path(
        r"tbo_residence_time_results"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    XY_GRID_CSV = OUTPUT_DIR / "xy_synthetic_residence_grid.csv"
    YZ_GRID_CSV = OUTPUT_DIR / "yz_synthetic_residence_grid.csv"

    # ------------------------------------------------------------
    # Column names
    # ------------------------------------------------------------

    OBJECT_ID_COL = "Object id"
    EPOCH_COL = "Julian Date"

    X_COL = "Synodic x"
    Y_COL = "Synodic y"
    Z_COL = "Synodic z"

    # ------------------------------------------------------------
    # File filtering
    # ------------------------------------------------------------

    TRAJECTORY_FILE_GLOB = "*.csv"

    EXCLUDE_FILE_KEYWORDS = [
        "MASTER",
        "master",
        "summary",
        "window",
        "sweep",
        "membership",
        "residence_grid",
        "synthetic_residence",
    ]

    # ------------------------------------------------------------
    # Binning in AU
    # ------------------------------------------------------------

    X_MIN, X_MAX = -0.04, 0.04
    Y_MIN, Y_MAX = -0.04, 0.04
    Z_MIN, Z_MAX = -0.04, 0.04

    NX = 300
    NY = 300
    NZ = 300

    x_edges = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, NY + 1)
    z_edges = np.linspace(Z_MIN, Z_MAX, NZ + 1)

    # ------------------------------------------------------------
    # Overlay options
    # ------------------------------------------------------------

    OVERLAY_REAL_TBOS = True

    # In plot_from_grid mode, this controls whether the script re-reads
    # only the real TBO trajectory files for overlay.
    LOAD_REAL_OVERLAY_IN_GRID_MODE = False

    REAL_OVERLAY_MAX_POINTS_PER_OBJECT = 3_000

    # ------------------------------------------------------------
    # Plotting options
    # ------------------------------------------------------------

    PROGRESS_EVERY = 100

    SIGMA = 1.2
    N_LEVELS = 30
    CMAP = "hot_r"

    USE_SHARED_COLOR_SCALE = False

    # ------------------------------------------------------------
    # Accumulate or load grids
    # ------------------------------------------------------------

    if RUN_MODE == "accumulate_and_plot":

        result = accumulate_synthetic_residence_time_maps(
            trajectory_folder=TRAJECTORY_FOLDER,
            object_id_col=OBJECT_ID_COL,
            epoch_col=EPOCH_COL,
            x_col=X_COL,
            y_col=Y_COL,
            z_col=Z_COL,
            x_edges=x_edges,
            y_edges=y_edges,
            z_edges=z_edges,
            trajectory_file_glob=TRAJECTORY_FILE_GLOB,
            progress_every=PROGRESS_EVERY,
            overlay_real_tbos=OVERLAY_REAL_TBOS,
            real_overlay_max_points_per_object=REAL_OVERLAY_MAX_POINTS_PER_OBJECT,
            exclude_file_keywords=EXCLUDE_FILE_KEYWORDS,
        )

        H_xy = result["H_xy"]
        H_yz = result["H_yz"]
        real_overlay = result["real_overlay"]

        print("\nAccumulation complete")
        print(f"Objects processed: {result['n_objects_processed']:,}")
        print(f"Synthetic objects used for map: {result['n_synthetic_used']:,}")
        print(f"Real TBOs collected for overlay: {result['n_real_overlay']:,}")
        print(
            f"Total synthetic residence time: "
            f"{result['total_synthetic_time_days']:,.2f} days"
        )

        save_histogram_csv(
            H_xy,
            x_edges,
            y_edges,
            x_name="Synodic x (AU)",
            y_name="Synodic y (AU)",
            output_csv=XY_GRID_CSV,
        )

        save_histogram_csv(
            H_yz,
            y_edges,
            z_edges,
            x_name="Synodic y (AU)",
            y_name="Synodic z (AU)",
            output_csv=YZ_GRID_CSV,
        )

        print(f"Wrote {XY_GRID_CSV}")
        print(f"Wrote {YZ_GRID_CSV}")

    elif RUN_MODE == "plot_from_grid":

        print("Loading residence grids from CSV...")

        H_xy, x_edges, y_edges = load_histogram_csv(
            grid_csv=XY_GRID_CSV,
            x_name="Synodic x (AU)",
            y_name="Synodic y (AU)",
            value_name="residence_time_days",
        )

        H_yz, y_edges, z_edges = load_histogram_csv(
            grid_csv=YZ_GRID_CSV,
            x_name="Synodic y (AU)",
            y_name="Synodic z (AU)",
            value_name="residence_time_days",
        )

        print(f"Loaded {XY_GRID_CSV}")
        print(f"Loaded {YZ_GRID_CSV}")

        if OVERLAY_REAL_TBOS and LOAD_REAL_OVERLAY_IN_GRID_MODE:
            real_overlay = load_real_tbo_overlays(
                trajectory_folder=TRAJECTORY_FOLDER,
                object_id_col=OBJECT_ID_COL,
                epoch_col=EPOCH_COL,
                x_col=X_COL,
                y_col=Y_COL,
                z_col=Z_COL,
                trajectory_file_glob=TRAJECTORY_FILE_GLOB,
                exclude_file_keywords=EXCLUDE_FILE_KEYWORDS,
                real_overlay_max_points_per_object=REAL_OVERLAY_MAX_POINTS_PER_OBJECT,
                progress_every=PROGRESS_EVERY,
            )
        else:
            real_overlay = []

    else:
        raise ValueError(
            "RUN_MODE must be either 'accumulate_and_plot' or 'plot_from_grid'"
        )

    # ------------------------------------------------------------
    # Shared color scale
    # ------------------------------------------------------------

    if USE_SHARED_COLOR_SCALE:
        H_xy_smooth = gaussian_filter(H_xy.T, sigma=SIGMA)
        H_yz_smooth = gaussian_filter(H_yz.T, sigma=SIGMA)

        shared_vmax = max(
            np.nanmax(H_xy_smooth),
            np.nanmax(H_yz_smooth),
        )
    else:
        shared_vmax = None

    # ------------------------------------------------------------
    # Plot XY projection
    # ------------------------------------------------------------

    plot_residence_contour(
        H=H_xy,
        x_edges=x_edges,
        y_edges=y_edges,
        xlabel="Synodic x (AU)",
        ylabel="Synodic y (AU)",
        title="Synthetic TBO Residence Time Distribution, XY Projection",
        output_path=OUTPUT_DIR / "xy_synthetic_residence_time_with_overlays.png",
        projection="xy",
        real_overlay=real_overlay,
        overlay_real=OVERLAY_REAL_TBOS,
        sigma=SIGMA,
        n_levels=N_LEVELS,
        cmap=CMAP,
        vmax=shared_vmax,
    )

    # ------------------------------------------------------------
    # Plot YZ projection
    # ------------------------------------------------------------

    plot_residence_contour(
        H=H_yz,
        x_edges=y_edges,
        y_edges=z_edges,
        xlabel="Synodic y (AU)",
        ylabel="Synodic z (AU)",
        title="Synthetic TBO Residence Time Distribution, YZ Projection",
        output_path=OUTPUT_DIR / "yz_synthetic_residence_time_with_overlays.png",
        projection="yz",
        real_overlay=real_overlay,
        overlay_real=OVERLAY_REAL_TBOS,
        sigma=SIGMA,
        n_levels=N_LEVELS,
        cmap=CMAP,
        vmax=shared_vmax,
    )

    print(f"\nWrote plots to: {OUTPUT_DIR}")
    print(f"1 EH radius: {ONE_EH_AU:.5f} AU")
    print(f"Halo y-radius: {HALO_Y_RADIUS_AU:.6f} AU")
    print(f"Halo z-radius: {HALO_Z_RADIUS_AU:.6f} AU")