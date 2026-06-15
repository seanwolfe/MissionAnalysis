from pathlib import Path
from typing import Iterable, Optional

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

    if object_id.upper().startswith("NESC"):
        return "synthetic"

    return "real"


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


# ============================================================
# Loading
# ============================================================

def load_master_file(
    master_csv: Path,
    object_id_col: str,
    h_col: str,
    diameter_col: str,
) -> pd.DataFrame:
    """
    Load the master file containing object-level H and D.
    """

    master = pd.read_csv(master_csv)

    required_cols = [object_id_col, h_col, diameter_col]
    missing = [c for c in required_cols if c not in master.columns]

    if missing:
        raise ValueError(f"Master file is missing required columns: {missing}")

    master = master[required_cols].copy()

    master[object_id_col] = master[object_id_col].apply(normalize_object_id)
    master[h_col] = pd.to_numeric(master[h_col], errors="coerce")
    master[diameter_col] = pd.to_numeric(master[diameter_col], errors="coerce")

    master["population"] = master[object_id_col].apply(infer_population_from_object_id)

    # If duplicate objects exist, keep the first.
    master = master.drop_duplicates(subset=[object_id_col], keep="first")

    return master


def load_trajectory_folder(
    trajectory_folder: Path,
    master: pd.DataFrame,
    object_id_col: str,
    epoch_col: str,
    distance_col: str,
    h_col: str,
    diameter_col: str,
    file_glob: str = "*.csv",
) -> pd.DataFrame:
    """
    Load one trajectory CSV per object and merge H, D, and population from master.

    If an individual trajectory CSV does not contain Object id, the file name stem
    is used as the object ID.
    """

    trajectory_folder = Path(trajectory_folder)
    files = sorted(trajectory_folder.glob(file_glob))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No trajectory files matching '{file_glob}' found in {trajectory_folder}"
        )

    dfs = []

    for file in files:
        try:
            df_i = pd.read_csv(file)
        except Exception as e:
            print(f"Skipping {file.name}: could not read CSV. Error: {e}")
            continue

        if object_id_col not in df_i.columns:
            df_i[object_id_col] = file.stem

        required_traj_cols = [object_id_col, epoch_col, distance_col]
        missing = [c for c in required_traj_cols if c not in df_i.columns]

        if missing:
            print(f"Skipping {file.name}: missing trajectory columns {missing}")
            continue

        df_i[object_id_col] = df_i[object_id_col].apply(normalize_object_id)

        df_i = df_i.merge(
            master[[object_id_col, h_col, diameter_col, "population"]],
            on=object_id_col,
            how="left",
            validate="many_to_one",
        )

        missing_master = df_i[df_i[h_col].isna() | df_i[diameter_col].isna()]
        if len(missing_master) > 0:
            missing_ids = missing_master[object_id_col].drop_duplicates().tolist()
            print(
                f"Warning: {file.name} has object ID(s) missing H or D in master: "
                f"{missing_ids[:5]}"
            )

        dfs.append(df_i)

    if len(dfs) == 0:
        raise RuntimeError("No valid trajectory CSV files were loaded.")

    df = pd.concat(dfs, ignore_index=True)

    df[epoch_col] = pd.to_numeric(df[epoch_col], errors="coerce")
    df[distance_col] = pd.to_numeric(df[distance_col], errors="coerce")
    df[h_col] = pd.to_numeric(df[h_col], errors="coerce")
    df[diameter_col] = pd.to_numeric(df[diameter_col], errors="coerce")

    df = df.dropna(
        subset=[
            object_id_col,
            epoch_col,
            distance_col,
            h_col,
            diameter_col,
        ]
    )

    return df


# ============================================================
# Population filtering
# ============================================================

def filter_population(
    df: pd.DataFrame,
    population_mode: str,
) -> pd.DataFrame:
    """
    population_mode options:
        "synthetic" -> Object id starts with NESC
        "real"      -> Object id does not start with NESC
        "both"      -> all objects
    """

    population_mode = population_mode.lower().strip()

    if population_mode == "both":
        return df.copy()

    if population_mode == "synthetic":
        return df[df["population"] == "synthetic"].copy()

    if population_mode == "real":
        return df[df["population"] == "real"].copy()

    raise ValueError(
        f"Invalid population_mode='{population_mode}'. "
        "Use 'synthetic', 'real', or 'both'."
    )


# ============================================================
# Windowing
# ============================================================

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


def summarize_window_membership(
    df_window: pd.DataFrame,
    object_id_col: str,
    epoch_col: str,
    distance_col: str,
    h_col: str,
    diameter_col: str,
) -> pd.DataFrame:
    """
    Convert qualifying epoch rows in one window into one row per object.
    """

    grouped = df_window.groupby(object_id_col, dropna=False)

    membership = grouped.agg(
        H=(h_col, "first"),
        D=(diameter_col, "first"),
        population=("population", "first"),
        first_epoch_in_window=(epoch_col, "min"),
        last_epoch_in_window=(epoch_col, "max"),
        n_samples=(epoch_col, "size"),
        min_distance=(distance_col, "min"),
        median_distance=(distance_col, "median"),
    ).reset_index()

    membership["duration_days"] = (
        membership["last_epoch_in_window"] - membership["first_epoch_in_window"]
    )

    return membership


# ============================================================
# Main analysis
# ============================================================

def sliding_detection_opportunities(
    df: pd.DataFrame,
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
    output_membership: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main sliding-window detection opportunity analysis.

    Returns:
        window_summary_df
        sweep_summary_df
        window_membership_df
    """

    required_cols = [
        object_id_col,
        epoch_col,
        distance_col,
        h_col,
        diameter_col,
        "population",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    df = df.copy()

    if start_epoch is None:
        start_epoch = float(df[epoch_col].min())

    if end_epoch is None:
        end_epoch = float(df[epoch_col].max())

    window_summary_rows = []
    membership_rows = []

    for population_mode in population_modes:
        df_pop = filter_population(df, population_mode)

        if len(df_pop) == 0:
            print(f"Warning: no rows found for population_mode='{population_mode}'")

        for window_length_days in window_lengths_days:
            starts = generate_window_starts(
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                window_length_days=window_length_days,
                window_step_days=window_step_days,
                include_partial_windows=include_partial_windows,
            )

            for window_start in starts:
                window_end = window_start + window_length_days

                in_window = (
                    (df_pop[epoch_col] >= window_start)
                    & (df_pop[epoch_col] < window_end)
                )

                in_distance = (
                    (df_pop[distance_col] >= distance_min)
                    & (df_pop[distance_col] <= distance_max)
                )

                df_qual = df_pop.loc[in_window & in_distance].copy()

                if len(df_qual) == 0:
                    stats_H = scalar_statistics(pd.Series(dtype=float), "H")
                    stats_D = scalar_statistics(pd.Series(dtype=float), "D")

                    window_summary_rows.append(
                        {
                            "population_mode": population_mode,
                            "window_length_days": window_length_days,
                            "window_step_days": window_step_days,
                            "window_start": window_start,
                            "window_end": window_end,
                            "distance_min": distance_min,
                            "distance_max": distance_max,
                            "n_objects": 0,
                            "n_samples": 0,
                            "duration_mean_days": np.nan,
                            "duration_median_days": np.nan,
                            **stats_H,
                            **stats_D,
                        }
                    )
                    continue

                membership = summarize_window_membership(
                    df_window=df_qual,
                    object_id_col=object_id_col,
                    epoch_col=epoch_col,
                    distance_col=distance_col,
                    h_col=h_col,
                    diameter_col=diameter_col,
                )

                if min_duration_days > 0:
                    membership = membership[
                        membership["duration_days"] >= min_duration_days
                    ].copy()

                if len(membership) == 0:
                    stats_H = scalar_statistics(pd.Series(dtype=float), "H")
                    stats_D = scalar_statistics(pd.Series(dtype=float), "D")

                    window_summary_rows.append(
                        {
                            "population_mode": population_mode,
                            "window_length_days": window_length_days,
                            "window_step_days": window_step_days,
                            "window_start": window_start,
                            "window_end": window_end,
                            "distance_min": distance_min,
                            "distance_max": distance_max,
                            "n_objects": 0,
                            "n_samples": 0,
                            "duration_mean_days": np.nan,
                            "duration_median_days": np.nan,
                            **stats_H,
                            **stats_D,
                        }
                    )
                    continue

                stats_H = scalar_statistics(membership["H"], "H")
                stats_D = scalar_statistics(membership["D"], "D")

                window_summary_rows.append(
                    {
                        "population_mode": population_mode,
                        "window_length_days": window_length_days,
                        "window_step_days": window_step_days,
                        "window_start": window_start,
                        "window_end": window_end,
                        "distance_min": distance_min,
                        "distance_max": distance_max,
                        "n_objects": int(len(membership)),
                        "n_samples": int(membership["n_samples"].sum()),
                        "duration_mean_days": membership["duration_days"].mean(),
                        "duration_median_days": membership["duration_days"].median(),
                        **stats_H,
                        **stats_D,
                    }
                )

                if output_membership:
                    membership = membership.rename(
                        columns={
                            object_id_col: "object_id",
                        }
                    )

                    membership.insert(0, "population_mode", population_mode)
                    membership.insert(1, "window_length_days", window_length_days)
                    membership.insert(2, "window_step_days", window_step_days)
                    membership.insert(3, "window_start", window_start)
                    membership.insert(4, "window_end", window_end)
                    membership.insert(5, "distance_min", distance_min)
                    membership.insert(6, "distance_max", distance_max)

                    membership_rows.append(membership)

    window_summary_df = pd.DataFrame(window_summary_rows)

    if len(window_summary_df) == 0:
        sweep_summary_df = pd.DataFrame()
    else:
        sweep_summary_df = (
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

                H_median_mean=("H_median", "mean"),
                H_median_min=("H_median", "min"),
                H_median_max=("H_median", "max"),
                H_mean_mean=("H_mean", "mean"),

                D_median_mean=("D_median", "mean"),
                D_median_min=("D_median", "min"),
                D_median_max=("D_median", "max"),
                D_mean_mean=("D_mean", "mean"),

                duration_mean_days=("duration_mean_days", "mean"),
                duration_median_days=("duration_median_days", "median"),
            )
            .reset_index()
        )

    if output_membership and len(membership_rows) > 0:
        window_membership_df = pd.concat(membership_rows, ignore_index=True)
    else:
        window_membership_df = pd.DataFrame()

    return window_summary_df, sweep_summary_df, window_membership_df


# ============================================================
# PyCharm configuration section
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------

    TRAJECTORY_FOLDER = Path(
        r"/path/to/folder/with/one_csv_per_object"
    )

    MASTER_CSV = Path(
        r"/path/to/master.csv"
    )

    OUTPUT_DIR = Path(
        r"/path/to/output_folder"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Column names
    # ------------------------------------------------------------

    OBJECT_ID_COL = "Object id"
    EPOCH_COL = "Julian Date"

    # Choose which distance column defines the opportunity.
    #
    # Options from your trajectory files include, for example:
    #   "Distance"
    #   "sunearthl1-ast-dist"
    #   "sun-ast-dist"
    #
    # Since you said au, distance_min and distance_max below should also be in au.
    DISTANCE_COL = "Distance"

    H_COL = "H"

    # Diameter column in the master file.
    # You said this is D in meters.
    DIAMETER_COL = "D"

    # ------------------------------------------------------------
    # Distance opportunity definition
    # ------------------------------------------------------------

    DISTANCE_MIN = 0.0
    DISTANCE_MAX = 0.03

    # Example:
    #   0.03 au is approximately 3 Earth Hill radii if that is your convention.
    #   You can also use 0.01 for approximately 1 Earth Hill radius.

    # ------------------------------------------------------------
    # Window sweep
    # ------------------------------------------------------------

    WINDOW_LENGTHS_DAYS = [
        30.0,
        90.0,
        180.0,
        365.25,
        730.5,
    ]

    # Window starts are separated by this step, regardless of window length.
    WINDOW_STEP_DAYS = 30.0

    # Leave as None to use min/max Julian Date from the loaded trajectory files.
    START_EPOCH = None
    END_EPOCH = None

    # If False, only windows fully contained in [START_EPOCH, END_EPOCH] are used.
    INCLUDE_PARTIAL_WINDOWS = False

    # ------------------------------------------------------------
    # Population modes
    # ------------------------------------------------------------

    # Object id starting with NESC -> synthetic.
    # Anything else -> real.
    #
    # Options:
    #   ["synthetic"]
    #   ["real"]
    #   ["both"]
    #   ["synthetic", "real", "both"]
    POPULATION_MODES = [
        "synthetic",
        "real",
        "both",
    ]

    # ------------------------------------------------------------
    # Duration filtering
    # ------------------------------------------------------------

    # If 0.0, a single qualifying epoch is enough for an object to count.
    #
    # If > 0.0, the object must remain in the distance range inside the window
    # for at least this estimated duration:
    #
    #   last qualifying epoch - first qualifying epoch
    #
    MIN_DURATION_DAYS = 0.0

    # ------------------------------------------------------------
    # Output options
    # ------------------------------------------------------------

    OUTPUT_MEMBERSHIP = True

    TRAJECTORY_FILE_GLOB = "*.csv"

    # ------------------------------------------------------------
    # Run
    # ------------------------------------------------------------

    master_df = load_master_file(
        master_csv=MASTER_CSV,
        object_id_col=OBJECT_ID_COL,
        h_col=H_COL,
        diameter_col=DIAMETER_COL,
    )

    trajectory_df = load_trajectory_folder(
        trajectory_folder=TRAJECTORY_FOLDER,
        master=master_df,
        object_id_col=OBJECT_ID_COL,
        epoch_col=EPOCH_COL,
        distance_col=DISTANCE_COL,
        h_col=H_COL,
        diameter_col=DIAMETER_COL,
        file_glob=TRAJECTORY_FILE_GLOB,
    )

    print(f"Loaded {len(trajectory_df):,} trajectory rows")
    print(f"Loaded {trajectory_df[OBJECT_ID_COL].nunique():,} unique objects")

    window_summary_df, sweep_summary_df, window_membership_df = (
        sliding_detection_opportunities(
            df=trajectory_df,
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
            output_membership=OUTPUT_MEMBERSHIP,
        )
    )

    window_summary_path = OUTPUT_DIR / "window_summary.csv"
    sweep_summary_path = OUTPUT_DIR / "sweep_summary.csv"
    membership_path = OUTPUT_DIR / "window_membership.csv"

    window_summary_df.to_csv(window_summary_path, index=False)
    sweep_summary_df.to_csv(sweep_summary_path, index=False)

    if OUTPUT_MEMBERSHIP:
        window_membership_df.to_csv(membership_path, index=False)

    print(f"Wrote {window_summary_path}")
    print(f"Wrote {sweep_summary_path}")

    if OUTPUT_MEMBERSHIP:
        print(f"Wrote {membership_path}")