"""
Plot ballistic proxy cost versus optimized total delta-v.

Designed to run directly from PyCharm:
1. Edit the USER CONFIGURATION section.
2. Press the green Run button.

The script:
- keeps only target-successful cases;
- requires finite ballistic_proxy_cost and dv_total_mps;
- removes duplicate injection states using the six-component initial state;
- represents each duplicate group by the median proxy cost and median delta-v;
- creates both scatter and hexbin plots;
- optionally uses logarithmic x and/or y axes;
- saves the figures and optionally displays them.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIGURATION
# =============================================================================

INPUT_CSV = Path(
    r"C:\Users\seanw\Desktop\MissionAnalysis\four_burn_optimized_sweep_summary.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "transfer\proxy_cost_vs_delta_v_plots"

# CSV settings
CSV_SEPARATOR = ","

# Axis scaling: use "linear" or "log"
X_SCALE = "linear"
Y_SCALE = "linear"

# Figure settings
SCATTER_POINT_SIZE = 10
SCATTER_ALPHA = 0.45
HEXBIN_GRID_SIZE = 70
HEXBIN_MIN_COUNT = 1
FIGURE_DPI = 300
FIGURE_SIZE = (8.0, 6.0)

# Save formats, for example ["png"], ["png", "pdf"], or ["pdf"]
SAVE_FORMATS = ["svg"]

SHOW_FIGURES = True

# Duplicate-state handling
#
# Two rows are treated as the same injection state when all three position
# components and all three velocity components fall in the same tolerance bins.
#
# These defaults are deliberately very strict:
#   1e-6 km  = 1 mm
#   1e-12 km/s = 1e-9 m/s
#
# Increase these only if the same state was written with small numerical noise.
POSITION_DUPLICATE_TOLERANCE_KM = 1.0e-6
VELOCITY_DUPLICATE_TOLERANCE_KMS = 1.0e-12

# If duplicate states have different successful optimization results, use the
# median proxy cost and median delta-v for the single plotted representative.
DUPLICATE_AGGREGATION = "median"  # supported: "median", "minimum_dv"

# =============================================================================
# END USER CONFIGURATION
# =============================================================================


STATE_COLUMNS = [
    "initial_x_geo_j2000_km",
    "initial_y_geo_j2000_km",
    "initial_z_geo_j2000_km",
    "initial_vx_geo_j2000_kms",
    "initial_vy_geo_j2000_kms",
    "initial_vz_geo_j2000_kms",
]

REQUIRED_COLUMNS = [
    "target_success",
    "ballistic_proxy_cost",
    "dv_total_mps",
    *STATE_COLUMNS,
]


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to a Boolean Series."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.eq(1).fillna(False)

    normalized = series.astype("string").str.strip().str.lower()

    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "nan", "<na>"}

    unknown = normalized[~normalized.isin(true_values | false_values)]
    if not unknown.empty:
        examples = ", ".join(map(str, unknown.dropna().unique()[:5]))
        warnings.warn(
            "Unrecognized target_success values were treated as False. "
            f"Examples: {examples}"
        )

    return normalized.isin(true_values)


def validate_configuration() -> None:
    if X_SCALE not in {"linear", "log"}:
        raise ValueError("X_SCALE must be either 'linear' or 'log'.")

    if Y_SCALE not in {"linear", "log"}:
        raise ValueError("Y_SCALE must be either 'linear' or 'log'.")

    if POSITION_DUPLICATE_TOLERANCE_KM <= 0.0:
        raise ValueError("POSITION_DUPLICATE_TOLERANCE_KM must be positive.")

    if VELOCITY_DUPLICATE_TOLERANCE_KMS <= 0.0:
        raise ValueError("VELOCITY_DUPLICATE_TOLERANCE_KMS must be positive.")

    if DUPLICATE_AGGREGATION not in {"median", "minimum_dv"}:
        raise ValueError(
            "DUPLICATE_AGGREGATION must be 'median' or 'minimum_dv'."
        )


def read_and_filter_data(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read required columns and retain finite target-successful cases."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found:\n{path}")

    header = pd.read_csv(path, sep=CSV_SEPARATOR, nrows=0)
    missing = [column for column in REQUIRED_COLUMNS if column not in header.columns]
    if missing:
        missing_text = "\n  - ".join(missing)
        raise ValueError(
            "The input CSV is missing required columns:\n"
            f"  - {missing_text}"
        )

    df = pd.read_csv(
        path,
        sep=CSV_SEPARATOR,
        usecols=REQUIRED_COLUMNS,
        low_memory=False,
    )

    counts: dict[str, int] = {"rows_read": len(df)}

    df["target_success"] = parse_boolean_series(df["target_success"])
    df = df.loc[df["target_success"]].copy()
    counts["target_success_rows"] = len(df)

    numeric_columns = ["ballistic_proxy_cost", "dv_total_mps", *STATE_COLUMNS]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    finite_mask = np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all(axis=1)
    counts["nonfinite_rows_removed"] = int((~finite_mask).sum())
    df = df.loc[finite_mask].copy()

    if X_SCALE == "log":
        positive_x = df["ballistic_proxy_cost"] > 0.0
        counts["nonpositive_proxy_rows_removed"] = int((~positive_x).sum())
        df = df.loc[positive_x].copy()
    else:
        counts["nonpositive_proxy_rows_removed"] = 0

    if Y_SCALE == "log":
        positive_y = df["dv_total_mps"] > 0.0
        counts["nonpositive_dv_rows_removed"] = int((~positive_y).sum())
        df = df.loc[positive_y].copy()
    else:
        counts["nonpositive_dv_rows_removed"] = 0

    counts["rows_before_deduplication"] = len(df)

    if df.empty:
        raise ValueError(
            "No rows remain after filtering for target-successful, finite cases."
        )

    return df, counts


def add_duplicate_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer quantization keys for duplicate-state identification."""
    result = df.copy()

    position_columns = STATE_COLUMNS[:3]
    velocity_columns = STATE_COLUMNS[3:]

    for column in position_columns:
        result[f"__key_{column}"] = np.rint(
            result[column].to_numpy(dtype=float)
            / POSITION_DUPLICATE_TOLERANCE_KM
        ).astype(np.int64)

    for column in velocity_columns:
        result[f"__key_{column}"] = np.rint(
            result[column].to_numpy(dtype=float)
            / VELOCITY_DUPLICATE_TOLERANCE_KMS
        ).astype(np.int64)

    return result


def collapse_duplicate_states(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    """Collapse duplicate injection states to one plotted representative."""
    keyed = add_duplicate_group_keys(df)
    key_columns = [f"__key_{column}" for column in STATE_COLUMNS]

    group_sizes = keyed.groupby(key_columns, sort=False).size()
    duplicate_groups = int((group_sizes > 1).sum())

    if DUPLICATE_AGGREGATION == "median":
        collapsed = (
            keyed.groupby(key_columns, sort=False, as_index=False)
            .agg(
                ballistic_proxy_cost=("ballistic_proxy_cost", "median"),
                dv_total_mps=("dv_total_mps", "median"),
                duplicate_count=("dv_total_mps", "size"),
            )
        )

    elif DUPLICATE_AGGREGATION == "minimum_dv":
        minimum_indices = (
            keyed.groupby(key_columns, sort=False)["dv_total_mps"].idxmin()
        )
        collapsed = keyed.loc[
            minimum_indices,
            ["ballistic_proxy_cost", "dv_total_mps"],
        ].copy()

        duplicate_counts = (
            keyed.groupby(key_columns, sort=False)
            .size()
            .to_numpy(dtype=int)
        )
        collapsed["duplicate_count"] = duplicate_counts

    else:
        raise RuntimeError("Unsupported duplicate aggregation mode.")

    duplicate_rows_removed = len(df) - len(collapsed)
    return collapsed, duplicate_rows_removed, duplicate_groups


def configure_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("Ballistic Cost $J_b$")
    ax.set_ylabel(r"Total $\Delta v$ (m/s)")
    ax.set_xscale(X_SCALE)
    ax.set_yscale(Y_SCALE)
    ax.grid(True, alpha=0.25)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for extension in SAVE_FORMATS:
        extension = extension.lower().lstrip(".")
        output_path = OUTPUT_DIR / f"{stem}.{extension}"
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {output_path}")


def make_scatter_plot(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.scatter(
        df["ballistic_proxy_cost"],
        df["dv_total_mps"],
        s=SCATTER_POINT_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
    )

    configure_axes(ax)
    fig.tight_layout()
    return fig


def make_hexbin_plot(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    hexbin = ax.hexbin(
        df["ballistic_proxy_cost"],
        df["dv_total_mps"],
        gridsize=HEXBIN_GRID_SIZE,
        mincnt=HEXBIN_MIN_COUNT,
        bins="log",
        xscale=X_SCALE,
        yscale=Y_SCALE,
    )

    configure_axes(ax)

    colourbar = fig.colorbar(hexbin, ax=ax)
    colourbar.set_label("Number of cases")

    fig.tight_layout()
    return fig


def print_summary(
    counts: dict[str, int],
    collapsed_rows: int,
    duplicate_rows_removed: int,
    duplicate_groups: int,
) -> None:
    print("\nProxy cost versus delta-v plot summary")
    print("-" * 44)
    print(f"Rows read:                         {counts['rows_read']:,}")
    print(f"Target-successful rows:            {counts['target_success_rows']:,}")
    print(
        "Rows removed for missing/nonfinite values: "
        f"{counts['nonfinite_rows_removed']:,}"
    )

    if X_SCALE == "log":
        print(
            "Rows removed for nonpositive proxy cost:  "
            f"{counts['nonpositive_proxy_rows_removed']:,}"
        )

    if Y_SCALE == "log":
        print(
            "Rows removed for nonpositive delta-v:     "
            f"{counts['nonpositive_dv_rows_removed']:,}"
        )

    print(
        "Rows before duplicate removal:     "
        f"{counts['rows_before_deduplication']:,}"
    )
    print(f"Duplicate-state groups:            {duplicate_groups:,}")
    print(f"Duplicate rows removed:             {duplicate_rows_removed:,}")
    print(f"Unique injection states plotted:    {collapsed_rows:,}")
    print(f"Duplicate aggregation:              {DUPLICATE_AGGREGATION}")
    print(f"Output directory:                   {OUTPUT_DIR}")


def main() -> None:
    validate_configuration()

    filtered, counts = read_and_filter_data(INPUT_CSV)
    collapsed, duplicate_rows_removed, duplicate_groups = (
        collapse_duplicate_states(filtered)
    )

    print_summary(
        counts=counts,
        collapsed_rows=len(collapsed),
        duplicate_rows_removed=duplicate_rows_removed,
        duplicate_groups=duplicate_groups,
    )

    scatter_figure = make_scatter_plot(collapsed)
    save_figure(scatter_figure, "proxy_cost_vs_delta_v_scatter")

    hexbin_figure = make_hexbin_plot(collapsed)
    save_figure(hexbin_figure, "proxy_cost_vs_delta_v_hexbin")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(scatter_figure)
        plt.close(hexbin_figure)


if __name__ == "__main__":
    main()
