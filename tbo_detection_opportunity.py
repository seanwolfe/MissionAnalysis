from pathlib import Path
from typing import Iterable, Optional
import time

import numpy as np
import pandas as pd


# ============================================================
# Basic utilities
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


def scalar_statistics(values: pd.Series, prefix: str) -> dict:
    """
    Compute min/median/mean/std/p25/p75/max for a scalar object property.

    Used for both H and D.
    """
    x = pd.to_numeric(values, errors="coerce").dropna()

    if len(x) == 0:
        return {
            f"{prefix}_min": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_min": x.min(),
        f"{prefix}_median": x.median(),
        f"{prefix}_mean": x.mean(),
        f"{prefix}_std": x.std(ddof=1) if len(x) > 1 else np.nan,
        f"{prefix}_p25": x.quantile(0.25),
        f"{prefix}_p75": x.quantile(0.75),
        f"{prefix}_max": x.max(),
    }


def read_space_separated_csv(path: Path, **kwargs) -> pd.DataFrame:
    """
    Read files written with pandas sep=' '.

    If your files were written with:
        df.to_csv(path, sep=' ', index=False)

    then column names containing spaces, such as "Object id" and "Julian Date",
    should usually be quoted automatically and readable with this function.
    """
    return pd.read_csv(
        path,
        sep=" ",
        engine="python",
        skipinitialspace=True,
        **kwargs,
    )


# ============================================================
# Master file
# ============================================================

def load_master_file(
    master_csv: Path,
    object_id_col: str,
    h_col: str,
    diameter_col: str,
) -> pd.DataFrame:
    """
    Load master file containing object-level H and D.
    """

    master = read_space_separated_csv(master_csv)

    required_cols = [object_id_col, h_col, diameter_col]
    missing = [c for c in required_cols if c not in master.columns]

    if missing:
        raise ValueError(
            f"Master file is missing columns {missing}.\n"
            f"Columns found were:\n{list(master.columns)}"
        )

    master = master[required_cols].copy()

    master[object_id_col] = master[object_id_col].apply(normalize_object_id)
    master[h_col] = pd.to_numeric(master[h_col], errors="coerce")
    master[diameter_col] = pd.to_numeric(master[diameter_col], errors="coerce")

    master["population"] = master[object_id_col].apply(infer_population_from_object_id)

    master = master.dropna(subset=[object_id_col, h_col, diameter_col])
    master = master.drop_duplicates(subset=[object_id_col], keep="first")

    return master


# ============================================================
# Window generation
# ============================================================

def find_global_epoch_range(
    trajectory_files: list[Path],
    epoch_col: str,
    chunksize: int,
    progress_every: int = 250,
) -> tuple[float, float]:
    """
    Lightweight pass through the files to find global min/max epoch.
    Only reads the epoch column.
    """

    global_min = np.inf
    global_max = -np.inf

    start_time_wall = time.time()
    n_files_total = len(trajectory_files)

    print("Scanning trajectory files to determine global epoch range...")

    for i, file in enumerate(trajectory_files, start=1):
        try:
            reader = read_space_separated_csv(
                file,
                usecols=[epoch_col],
                chunksize=chunksize,
            )

            for chunk in reader:
                epochs = pd.to_numeric(chunk[epoch_col], errors="coerce").dropna()

                if len(epochs) == 0:
                    continue

                global_min = min(global_min, epochs.min())
                global_max = max(global_max, epochs.max())

        except Exception as e:
            print(f"Warning: could not scan epochs in {file.name}. Error: {e}")

        if i == 1 or i == n_files_total or i % progress_every == 0:
            elapsed = time.time() - start_time_wall
            percent_done = 100.0 * i / n_files_total
            files_per_sec = i / elapsed if elapsed > 0 else np.nan
            remaining_files = n_files_total - i
            eta_sec = remaining_files / files_per_sec if files_per_sec > 0 else np.nan

            print(
                f"[epoch scan {i:,}/{n_files_total:,}] "
                f"{percent_done:6.2f}% complete | "
                f"elapsed: {elapsed / 60:.1f} min | "
                f"ETA: {eta_sec / 60:.1f} min"
            )

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("Could not determine global epoch range from trajectory files.")

    return float(global_min), float(global_max)


def generate_window_starts(
    start_epoch: float,
    end_epoch: float,
    window_length_days: float,
    window_step_days: float,
    include_partial_windows: bool = False,
) -> np.ndarray:
    """
    Generate sliding-window start epochs.

    Complete-window mode:
        window_start + window_length_days <= end_epoch

    Partial-window mode:
        window_start <= end_epoch
    """

    starts = []
    current = float(start_epoch)

    if include_partial_windows:
        while current <= end_epoch:
            starts.append(current)
            current += window_step_days
    else:
        while current + window_length_days <= end_epoch + 1e-12:
            starts.append(current)
            current += window_step_days

    return np.array(starts, dtype=float)


def make_window_grid(
    population_modes: Iterable[str],
    window_lengths_days: Iterable[float],
    window_step_days: float,
    start_epoch: float,
    end_epoch: float,
    distance_min: float,
    distance_max: float,
    include_partial_windows: bool,
) -> pd.DataFrame:
    """
    Make a complete grid of all requested windows.

    This lets the final window_summary.csv include zero-count windows too.
    """

    rows = []

    for population_mode in population_modes:
        for window_length_days in window_lengths_days:
            starts = generate_window_starts(
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                window_length_days=window_length_days,
                window_step_days=window_step_days,
                include_partial_windows=include_partial_windows,
            )

            for window_start in starts:
                rows.append(
                    {
                        "population_mode": population_mode,
                        "window_length_days": float(window_length_days),
                        "window_step_days": float(window_step_days),
                        "window_start": float(window_start),
                        "window_end": float(window_start + window_length_days),
                        "distance_min": float(distance_min),
                        "distance_max": float(distance_max),
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# Streaming trajectory processing
# ============================================================

def get_file_columns(file: Path) -> list[str]:
    """
    Read only the header to get column names.
    """
    header = read_space_separated_csv(file, nrows=0)
    return list(header.columns)


def process_one_trajectory_file(
    file: Path,
    master_lookup: dict,
    object_id_col: str,
    epoch_col: str,
    distance_col: str,
    h_col: str,
    diameter_col: str,
    distance_min: float,
    distance_max: float,
    population_modes: Iterable[str],
    window_starts_by_length: dict[float, np.ndarray],
    window_step_days: float,
    min_duration_days: float,
    chunksize: int,
) -> list[dict]:
    """
    Process one trajectory file and return compact membership rows.

    Only qualifying epochs inside the distance range are retained.
    """

    try:
        columns = get_file_columns(file)
    except Exception as e:
        print(f"Skipping {file.name}: could not read header. Error: {e}")
        return []

    has_object_id_col = object_id_col in columns

    usecols = [epoch_col, distance_col]

    if has_object_id_col:
        usecols.append(object_id_col)

    missing = [c for c in [epoch_col, distance_col] if c not in columns]

    if missing:
        print(
            f"Skipping {file.name}: missing columns {missing}. "
            f"Columns found were: {columns}"
        )
        return []

    qualifying_chunks = []

    try:
        reader = read_space_separated_csv(
            file,
            usecols=usecols,
            chunksize=chunksize,
        )

        for chunk in reader:
            if has_object_id_col:
                chunk[object_id_col] = chunk[object_id_col].apply(normalize_object_id)
            else:
                chunk[object_id_col] = file.stem

            chunk[epoch_col] = pd.to_numeric(chunk[epoch_col], errors="coerce")
            chunk[distance_col] = pd.to_numeric(chunk[distance_col], errors="coerce")

            chunk = chunk.dropna(subset=[object_id_col, epoch_col, distance_col])

            chunk = chunk[
                (chunk[distance_col] >= distance_min)
                & (chunk[distance_col] <= distance_max)
            ]

            if len(chunk) > 0:
                qualifying_chunks.append(
                    chunk[[object_id_col, epoch_col, distance_col]]
                )

    except Exception as e:
        print(f"Skipping {file.name}: could not process. Error: {e}")
        return []

    if len(qualifying_chunks) == 0:
        return []

    # This is usually small because it only includes distance-qualified epochs.
    dfq = pd.concat(qualifying_chunks, ignore_index=True)

    membership_rows = []

    # Normally one file = one object, but this handles accidental multi-object files.
    for object_id, df_obj in dfq.groupby(object_id_col, dropna=False):
        object_id = normalize_object_id(object_id)

        if object_id not in master_lookup:
            print(f"Warning: {file.name}: object ID '{object_id}' not found in master.")
            continue

        H = master_lookup[object_id][h_col]
        D = master_lookup[object_id][diameter_col]
        population = master_lookup[object_id]["population"]

        df_obj = df_obj.sort_values(epoch_col)

        epochs = df_obj[epoch_col].to_numpy(dtype=float)
        distances = df_obj[distance_col].to_numpy(dtype=float)

        object_population_modes = ["both", population]

        for population_mode in population_modes:
            if population_mode not in object_population_modes:
                continue

            for window_length_days, starts in window_starts_by_length.items():
                if len(starts) == 0:
                    continue

                ends = starts + window_length_days

                # For each window, find qualifying epochs in [start, end).
                left_indices = np.searchsorted(epochs, starts, side="left")
                right_indices = np.searchsorted(epochs, ends, side="left")

                valid_windows = np.where(right_indices > left_indices)[0]

                for idx in valid_windows:
                    i0 = left_indices[idx]
                    i1 = right_indices[idx]

                    epoch_slice = epochs[i0:i1]
                    distance_slice = distances[i0:i1]

                    first_epoch = float(epoch_slice[0])
                    last_epoch = float(epoch_slice[-1])
                    duration_days = last_epoch - first_epoch

                    if duration_days < min_duration_days:
                        continue

                    membership_rows.append(
                        {
                            "population_mode": population_mode,
                            "window_length_days": float(window_length_days),
                            "window_step_days": float(window_step_days),
                            "window_start": float(starts[idx]),
                            "window_end": float(ends[idx]),
                            "distance_min": float(distance_min),
                            "distance_max": float(distance_max),
                            "object_id": object_id,
                            "population": population,
                            "H": float(H),
                            "D": float(D),
                            "first_epoch_in_window": first_epoch,
                            "last_epoch_in_window": last_epoch,
                            "n_samples": int(len(epoch_slice)),
                            "duration_days": float(duration_days),
                            "min_distance": float(np.min(distance_slice)),
                            "median_distance": float(np.median(distance_slice)),
                        }
                    )

    return membership_rows


# ============================================================
# Summary tables
# ============================================================

def build_window_summary(
    window_grid: pd.DataFrame,
    membership_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per population mode, window length, and window start.

    Includes zero-count windows.
    """

    key_cols = [
        "population_mode",
        "window_length_days",
        "window_step_days",
        "window_start",
        "window_end",
        "distance_min",
        "distance_max",
    ]

    empty_stats = {
        **scalar_statistics(pd.Series(dtype=float), "H"),
        **scalar_statistics(pd.Series(dtype=float), "D"),
    }

    if len(membership_df) == 0:
        out = window_grid.copy()
        out["n_objects"] = 0
        out["n_samples"] = 0
        out["duration_mean_days"] = np.nan
        out["duration_median_days"] = np.nan

        for key, val in empty_stats.items():
            out[key] = val

        return out

    grouped = membership_df.groupby(key_cols, dropna=False)

    rows = []

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(key_cols, keys))

        H_stats = scalar_statistics(g["H"], "H")
        D_stats = scalar_statistics(g["D"], "D")

        row = {
            **base,
            "n_objects": int(g["object_id"].nunique()),
            "n_samples": int(g["n_samples"].sum()),
            "duration_mean_days": g["duration_days"].mean(),
            "duration_median_days": g["duration_days"].median(),
            **H_stats,
            **D_stats,
        }

        rows.append(row)

    summary_nonzero = pd.DataFrame(rows)

    window_summary = window_grid.merge(
        summary_nonzero,
        on=key_cols,
        how="left",
    )

    window_summary["n_objects"] = window_summary["n_objects"].fillna(0).astype(int)
    window_summary["n_samples"] = window_summary["n_samples"].fillna(0).astype(int)

    return window_summary


def build_sweep_summary(window_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Condensed one-row-per-window-length summary.
    """

    if len(window_summary_df) == 0:
        return pd.DataFrame()

    sweep_summary = (
        window_summary_df
        .groupby(["population_mode", "window_length_days"], dropna=False)
        .agg(
            n_windows=("n_objects", "size"),

            n_objects_min=("n_objects", "min"),
            n_objects_median=("n_objects", "median"),
            n_objects_mean=("n_objects", "mean"),
            n_objects_max=("n_objects", "max"),

            n_samples_median=("n_samples", "median"),
            n_samples_mean=("n_samples", "mean"),

            H_min_mean=("H_min", "mean"),
            H_median_mean=("H_median", "mean"),
            H_median_min=("H_median", "min"),
            H_median_max=("H_median", "max"),
            H_mean_mean=("H_mean", "mean"),

            D_min_mean=("D_min", "mean"),
            D_median_mean=("D_median", "mean"),
            D_median_min=("D_median", "min"),
            D_median_max=("D_median", "max"),
            D_mean_mean=("D_mean", "mean"),

            duration_mean_days=("duration_mean_days", "mean"),
            duration_median_days=("duration_median_days", "median"),
        )
        .reset_index()
    )

    return sweep_summary


# ============================================================
# Main driver
# ============================================================

def run_streaming_detection_analysis(
    trajectory_folder: Path,
    master_csv: Path,
    output_dir: Path,
    object_id_col: str,
    epoch_col: str,
    distance_col: str,
    h_col: str,
    diameter_col: str,
    distance_min: float,
    distance_max: float,
    window_lengths_days: Iterable[float],
    window_step_days: float,
    population_modes: Iterable[str],
    start_epoch: Optional[float] = None,
    end_epoch: Optional[float] = None,
    include_partial_windows: bool = False,
    min_duration_days: float = 0.0,
    trajectory_file_glob: str = "*.csv",
    chunksize: int = 200_000,
    output_membership: bool = True,
    progress_every: int = 25,
    epoch_scan_progress_every: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    trajectory_folder = Path(trajectory_folder)
    master_csv = Path(master_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_files = sorted(trajectory_folder.glob(trajectory_file_glob))

    if len(trajectory_files) == 0:
        raise FileNotFoundError(
            f"No files matching {trajectory_file_glob} found in {trajectory_folder}"
        )

    print(f"Found {len(trajectory_files):,} trajectory files")

    master = load_master_file(
        master_csv=master_csv,
        object_id_col=object_id_col,
        h_col=h_col,
        diameter_col=diameter_col,
    )

    print(f"Loaded {len(master):,} objects from master file")

    master_lookup = (
        master
        .set_index(object_id_col)[[h_col, diameter_col, "population"]]
        .to_dict(orient="index")
    )

    if start_epoch is None or end_epoch is None:
        detected_start, detected_end = find_global_epoch_range(
            trajectory_files=trajectory_files,
            epoch_col=epoch_col,
            chunksize=chunksize,
            progress_every=epoch_scan_progress_every,
        )

        if start_epoch is None:
            start_epoch = detected_start

        if end_epoch is None:
            end_epoch = detected_end

    print(f"Analysis epoch range: {start_epoch} to {end_epoch}")

    window_grid = make_window_grid(
        population_modes=population_modes,
        window_lengths_days=window_lengths_days,
        window_step_days=window_step_days,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        distance_min=distance_min,
        distance_max=distance_max,
        include_partial_windows=include_partial_windows,
    )

    print(f"Generated {len(window_grid):,} total windows")

    window_starts_by_length = {
        float(L): generate_window_starts(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            window_length_days=float(L),
            window_step_days=window_step_days,
            include_partial_windows=include_partial_windows,
        )
        for L in window_lengths_days
    }

    print("Window counts by length:")
    for L, starts in window_starts_by_length.items():
        print(f"  {L:g} days: {len(starts):,} windows")

    all_membership_rows = []

    start_time_wall = time.time()

    n_files_total = len(trajectory_files)
    n_files_processed = 0
    n_files_with_matches = 0
    n_membership_rows_total = 0

    print("Processing trajectory files...")

    for i, file in enumerate(trajectory_files, start=1):
        rows = process_one_trajectory_file(
            file=file,
            master_lookup=master_lookup,
            object_id_col=object_id_col,
            epoch_col=epoch_col,
            distance_col=distance_col,
            h_col=h_col,
            diameter_col=diameter_col,
            distance_min=distance_min,
            distance_max=distance_max,
            population_modes=population_modes,
            window_starts_by_length=window_starts_by_length,
            window_step_days=window_step_days,
            min_duration_days=min_duration_days,
            chunksize=chunksize,
        )

        n_files_processed += 1

        if rows:
            all_membership_rows.extend(rows)
            n_files_with_matches += 1
            n_membership_rows_total += len(rows)

        if i == 1 or i == n_files_total or i % progress_every == 0:
            elapsed = time.time() - start_time_wall
            files_per_sec = n_files_processed / elapsed if elapsed > 0 else np.nan

            remaining_files = n_files_total - n_files_processed
            eta_sec = remaining_files / files_per_sec if files_per_sec > 0 else np.nan

            percent_done = 100.0 * n_files_processed / n_files_total

            print(
                f"[{n_files_processed:,}/{n_files_total:,} objects] "
                f"{percent_done:6.2f}% complete | "
                f"current: {file.name} | "
                f"objects with opportunities: {n_files_with_matches:,} | "
                f"membership rows: {n_membership_rows_total:,} | "
                f"elapsed: {elapsed / 60:.1f} min | "
                f"ETA: {eta_sec / 60:.1f} min"
            )

    if all_membership_rows:
        membership_df = pd.DataFrame(all_membership_rows)
    else:
        membership_df = pd.DataFrame()

    print("Building window summary...")
    window_summary_df = build_window_summary(
        window_grid=window_grid,
        membership_df=membership_df,
    )

    print("Building sweep summary...")
    sweep_summary_df = build_sweep_summary(window_summary_df)

    window_summary_path = output_dir / "window_summary.csv"
    sweep_summary_path = output_dir / "sweep_summary.csv"
    membership_path = output_dir / "window_membership.csv"

    window_summary_df.to_csv(window_summary_path, index=False)
    sweep_summary_df.to_csv(sweep_summary_path, index=False)

    if output_membership:
        membership_df.to_csv(membership_path, index=False)

    total_elapsed = time.time() - start_time_wall

    print("\nDone.")
    print(f"Total processing time: {total_elapsed / 60:.1f} min")
    print(f"Objects processed: {n_files_processed:,}")
    print(f"Objects with opportunities: {n_files_with_matches:,}")
    print(f"Membership rows: {n_membership_rows_total:,}")
    print(f"Wrote {window_summary_path}")
    print(f"Wrote {sweep_summary_path}")

    if output_membership:
        print(f"Wrote {membership_path}")

    return window_summary_df, sweep_summary_df, membership_df


# ============================================================
# PyCharm configuration
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------

    TRAJECTORY_FOLDER = Path(
        r"/media/aeromec/Seagate Desktop Drive/minimoon_files_oorb"
    )

    MASTER_CSV = Path(
        r"/media/aeromec/Seagate Desktop Drive/minimoon_files_oorb/minimoon_master_with_L1_geo_omega_w_earth.csv"
    )

    OUTPUT_DIR = Path(
        r"tbo_window_results"
    )

    # ------------------------------------------------------------
    # Column names
    # ------------------------------------------------------------

    OBJECT_ID_COL = "Object id"
    EPOCH_COL = "Julian Date"

    # Choose the distance column used for the opportunity definition.
    #
    # Available examples from your trajectory files:
    #   "Distance"
    #   "sunearthl1-ast-dist"
    #   "sun-ast-dist"
    #
    # Units should match DISTANCE_MIN and DISTANCE_MAX.
    DISTANCE_COL = "Distance"

    # H and D come from the master file.
    H_COL = "H"
    DIAMETER_COL = "D"

    # ------------------------------------------------------------
    # Detection opportunity definition
    # ------------------------------------------------------------

    DISTANCE_MIN = 0.0
    DISTANCE_MAX = 0.03

    # ------------------------------------------------------------
    # Sliding-window sweep
    # ------------------------------------------------------------

    WINDOW_LENGTHS_DAYS = [
        180.0,
        365.25,
        730.5,
        365.25 * 3,
        365.25 * 4,
        365.25 * 5,
    ]

    # Window start spacing, independent of window length.
    WINDOW_STEP_DAYS = 180.0

    # Leave these as None to infer from all trajectory files.
    # If you already know the date range, setting these manually avoids
    # the initial epoch-scan pass and saves time.
    START_EPOCH = None
    END_EPOCH = None

    # If False, only complete windows are included.
    # If True, windows that extend past END_EPOCH are included.
    INCLUDE_PARTIAL_WINDOWS = False

    # ------------------------------------------------------------
    # Population modes
    # ------------------------------------------------------------

    # Object IDs beginning with NESC are synthetic.
    # All other object IDs are real.
    #
    # Options:
    #   ["synthetic"]
    #   ["real"]
    #   ["both"]
    #   ["synthetic", "real", "both"]
    POPULATION_MODES = [
        "synthetic"
    ]

    # ------------------------------------------------------------
    # Duration filter
    # ------------------------------------------------------------

    # 0.0 means a single qualifying epoch is enough.
    # If set to e.g. 7.0, the object must have at least 7 days between
    # first and last qualifying epoch inside the window.
    MIN_DURATION_DAYS = 0.0

    # ------------------------------------------------------------
    # Memory and progress settings
    # ------------------------------------------------------------

    TRAJECTORY_FILE_GLOB = "*.csv"

    # Number of rows read at a time from each trajectory file.
    # Increase for speed, decrease for lower memory.
    CHUNKSIZE = 200_000

    # Print progress every this many objects.
    PROGRESS_EVERY = 25

    # During the initial epoch scan, print progress every this many files.
    EPOCH_SCAN_PROGRESS_EVERY = 250

    # Whether to write the detailed per-object per-window membership file.
    OUTPUT_MEMBERSHIP = True

    # ------------------------------------------------------------
    # Run
    # ------------------------------------------------------------

    window_summary_df, sweep_summary_df, membership_df = run_streaming_detection_analysis(
        trajectory_folder=TRAJECTORY_FOLDER,
        master_csv=MASTER_CSV,
        output_dir=OUTPUT_DIR,
        object_id_col=OBJECT_ID_COL,
        epoch_col=EPOCH_COL,
        distance_col=DISTANCE_COL,
        h_col=H_COL,
        diameter_col=DIAMETER_COL,
        distance_min=DISTANCE_MIN,
        distance_max=DISTANCE_MAX,
        window_lengths_days=WINDOW_LENGTHS_DAYS,
        window_step_days=WINDOW_STEP_DAYS,
        population_modes=POPULATION_MODES,
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
        include_partial_windows=INCLUDE_PARTIAL_WINDOWS,
        min_duration_days=MIN_DURATION_DAYS,
        trajectory_file_glob=TRAJECTORY_FILE_GLOB,
        chunksize=CHUNKSIZE,
        output_membership=OUTPUT_MEMBERSHIP,
        progress_every=PROGRESS_EVERY,
        epoch_scan_progress_every=EPOCH_SCAN_PROGRESS_EVERY,
    )