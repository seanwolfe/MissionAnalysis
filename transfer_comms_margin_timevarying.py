"""
Plot time-varying link margin for two mothership LGAs and three DSN stations.

This script creates four separate figures:
    1. LGA 1 uplink margin for Goldstone, Canberra, and Madrid
    2. LGA 1 downlink margin for Goldstone, Canberra, and Madrid
    3. LGA 2 uplink margin for Goldstone, Canberra, and Madrid
    4. LGA 2 downlink margin for Goldstone, Canberra, and Madrid

Expected MATLAB-exported columns:
    Time
    Distance (km)
    Elevation (deg)
    EIRP (dBW)
    FSPL (dB)
    RIP (dB)
    C/No (dB-Hz)
    Eb/No (dB)
    Margin (dB)

Embedded line breaks in MATLAB column headers are handled automatically.

Run directly in PyCharm. Edit the CONFIGURATION section before running.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

# LGA 1 uplink: DSN transmitter -> mothership LGA 1 receiver
UPLINK_LGA1_GOLDSTONE_XLSX = Path(
    r"link_1_uplink_lga1_goldstone.xlsx"
)
UPLINK_LGA1_CANBERRA_XLSX = Path(
    r"link_3_uplink_lga1_canberra.xlsx"
)
UPLINK_LGA1_MADRID_XLSX = Path(
    r"link_5_uplink_lga1_madrid.xlsx"
)

# LGA 1 downlink: mothership LGA 1 transmitter -> DSN receiver
DOWNLINK_LGA1_GOLDSTONE_XLSX = Path(
    r"link_7_downlink_lga1_goldstone.xlsx"
)
DOWNLINK_LGA1_CANBERRA_XLSX = Path(
    r"link_9_downlink_lga1_canberra.xlsx"
)
DOWNLINK_LGA1_MADRID_XLSX = Path(
    r"link_11_downlink_lga1_madrid.xlsx"
)

# LGA 2 uplink: DSN transmitter -> mothership LGA 2 receiver
UPLINK_LGA2_GOLDSTONE_XLSX = Path(
    r"link_2_uplink_lga2_goldstone.xlsx"
)
UPLINK_LGA2_CANBERRA_XLSX = Path(
    r"link_4_uplink_lga2_canberra.xlsx"
)
UPLINK_LGA2_MADRID_XLSX = Path(
    r"link_6_uplink_lga2_madrid.xlsx"
)

# LGA 2 downlink: mothership LGA 2 transmitter -> DSN receiver
DOWNLINK_LGA2_GOLDSTONE_XLSX = Path(
    r"link_8_downlink_lga2_goldstone.xlsx"
)
DOWNLINK_LGA2_CANBERRA_XLSX = Path(
    r"link_10_downlink_lga2_canberra.xlsx"
)
DOWNLINK_LGA2_MADRID_XLSX = Path(
    r"link_12_downlink_lga2_madrid.xlsx"
)

# Sheet containing the link-budget table.
# Use 0 for the first sheet, or provide a sheet name such as "Sheet1".
EXCEL_SHEET_NAME: int | str = 0

OUTPUT_DIRECTORY = Path(r"dsn_lga_margin_plots")

# "elapsed_days" plots time since the common mission start.
# "datetime" plots absolute UTC.
X_AXIS_MODE = "elapsed_days"

# Leave as None to use the earliest timestamp among all 12 files.
# Example: MISSION_START_UTC = "2026-08-19 20:55:00"
MISSION_START_UTC: str | None = None

FIGURE_SIZE_INCHES = (9.0, 5.0)
OUTPUT_DPI = 300

SAVE_PNG = True
SAVE_SVG = True
SAVE_PDF = False
SHOW_FIGURES = True

# Example: Y_LIMITS_DB = (-20.0, 20.0)
Y_LIMITS_DB: tuple[float, float] | None = None

DRAW_ZERO_MARGIN_LINE = True

# Set to 1 to plot every row. Set to n > 1 to plot every nth row.
PLOT_EVERY_NTH_ROW = 1

# =============================================================================


DSN_LABELS = {
    "goldstone": "Goldstone DSS-26",
    "canberra": "Canberra DSS-34",
    "madrid": "Madrid DSS-55",
}


def normalize_column_name(column_name: object) -> str:
    """Normalize multiline Excel column headings to snake case."""
    name = str(column_name).replace("\ufeff", "").strip().lower()
    name = name.replace("\r", " ").replace("\n", " ")
    name = name.replace("c/no", "c_no")
    name = name.replace("eb/no", "eb_no")
    name = re.sub(r"[()]", " ", name)
    name = re.sub(r"[/\\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def parse_time_column(series: pd.Series, workbook_path: Path) -> pd.Series:
    """
    Parse Excel datetimes, timestamp strings, or numeric Excel serial dates.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    elif pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_datetime(
            series,
            unit="D",
            origin="1899-12-30",
            errors="coerce",
            utc=True,
        )
    else:
        parsed = pd.to_datetime(
            series.astype("string").str.strip(),
            errors="coerce",
            utc=True,
            format="mixed",
        )

    if parsed.notna().sum() == 0:
        raise ValueError(
            f"Could not parse any values in the Time column:\n{workbook_path}"
        )

    return parsed


def read_link_budget_workbook(workbook_path: Path) -> pd.DataFrame:
    """Read and clean Time and Margin columns from one XLSX workbook."""
    if not workbook_path.is_file():
        raise FileNotFoundError(
            f"Excel workbook does not exist:\n{workbook_path}\n\n"
            "Update the corresponding XLSX path in the CONFIGURATION section."
        )

    if workbook_path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError(
            f"Expected an .xlsx or .xlsm workbook, received:\n{workbook_path}"
        )

    try:
        dataframe = pd.read_excel(
            workbook_path,
            sheet_name=EXCEL_SHEET_NAME,
            engine="openpyxl",
        )
    except ImportError as exception:
        raise ImportError(
            "Reading XLSX files requires openpyxl.\n"
            "Install it in the PyCharm environment with:\n\n"
            "    pip install openpyxl\n"
        ) from exception

    dataframe.columns = [
        normalize_column_name(column) for column in dataframe.columns
    ]

    required_columns = {"time", "margin_db"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required column(s) {sorted(missing_columns)} in:\n"
            f"{workbook_path}\n\n"
            f"Normalized columns found:\n{list(dataframe.columns)}"
        )

    cleaned = dataframe.loc[:, ["time", "margin_db"]].copy()

    cleaned["time"] = parse_time_column(
        cleaned["time"],
        workbook_path,
    )
    cleaned["margin_db"] = pd.to_numeric(
        cleaned["margin_db"],
        errors="coerce",
    )

    cleaned = cleaned.dropna(subset=["time", "margin_db"])
    cleaned = cleaned.sort_values("time")
    cleaned = cleaned.drop_duplicates(subset="time", keep="last")
    cleaned = cleaned.reset_index(drop=True)

    if cleaned.empty:
        raise ValueError(
            f"No valid Time/Margin rows remain after cleaning:\n{workbook_path}"
        )

    if PLOT_EVERY_NTH_ROW > 1:
        cleaned = cleaned.iloc[::PLOT_EVERY_NTH_ROW].copy()

    print(
        f"Read {workbook_path.name}: "
        f"{len(cleaned):,} valid Time/Margin rows"
    )

    return cleaned


def build_file_groups() -> dict[str, dict[str, Path]]:
    """Define the four requested plot groups."""
    return {
        "uplink_lga1": {
            "goldstone": UPLINK_LGA1_GOLDSTONE_XLSX,
            "canberra": UPLINK_LGA1_CANBERRA_XLSX,
            "madrid": UPLINK_LGA1_MADRID_XLSX,
        },
        "downlink_lga1": {
            "goldstone": DOWNLINK_LGA1_GOLDSTONE_XLSX,
            "canberra": DOWNLINK_LGA1_CANBERRA_XLSX,
            "madrid": DOWNLINK_LGA1_MADRID_XLSX,
        },
        "uplink_lga2": {
            "goldstone": UPLINK_LGA2_GOLDSTONE_XLSX,
            "canberra": UPLINK_LGA2_CANBERRA_XLSX,
            "madrid": UPLINK_LGA2_MADRID_XLSX,
        },
        "downlink_lga2": {
            "goldstone": DOWNLINK_LGA2_GOLDSTONE_XLSX,
            "canberra": DOWNLINK_LGA2_CANBERRA_XLSX,
            "madrid": DOWNLINK_LGA2_MADRID_XLSX,
        },
    }


def determine_mission_start(
    loaded_data: dict[str, dict[str, pd.DataFrame]],
) -> pd.Timestamp:
    """Return the common origin for elapsed-time plots."""
    if MISSION_START_UTC is not None:
        return pd.to_datetime(MISSION_START_UTC, errors="raise", utc=True)

    first_times = [
        dataframe["time"].iloc[0]
        for group_data in loaded_data.values()
        for dataframe in group_data.values()
    ]
    return min(first_times)


def configure_datetime_axis(axis: plt.Axes) -> None:
    """Use concise automatic UTC date formatting."""
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    """Save one figure in the configured formats."""
    if SAVE_PNG:
        figure.savefig(
            output_stem.with_suffix(".png"),
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
        )
    if SAVE_SVG:
        figure.savefig(
            output_stem.with_suffix(".svg"),
            bbox_inches="tight",
        )
    if SAVE_PDF:
        figure.savefig(
            output_stem.with_suffix(".pdf"),
            bbox_inches="tight",
        )


def plot_margin_group(
    group_data: dict[str, pd.DataFrame],
    title: str,
    output_filename_stem: str,
    mission_start: pd.Timestamp,
) -> None:
    """Plot the three DSN margin histories for one LGA/link direction."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE_INCHES)

    for station_key in ("goldstone", "canberra", "madrid"):
        dataframe = group_data[station_key]

        if X_AXIS_MODE == "elapsed_days":
            x_values = (
                dataframe["time"] - mission_start
            ).dt.total_seconds() / 86400.0
        elif X_AXIS_MODE == "datetime":
            x_values = dataframe["time"]
        else:
            raise ValueError(
                'X_AXIS_MODE must be "elapsed_days" or "datetime".'
            )

        axis.scatter(
            x_values,
            dataframe["margin_db"],
            label=DSN_LABELS[station_key],
            s=5,
        )

    if DRAW_ZERO_MARGIN_LINE:
        axis.axhline(
            0.0,
            linewidth=1.0,
            linestyle="--",
            label="Zero-margin threshold",
        )

    axis.set_title(title)

    if X_AXIS_MODE == "elapsed_days":
        axis.set_xlabel("Time since departure (days)")
    else:
        axis.set_xlabel("Time (UTC)")
        configure_datetime_axis(axis)

    axis.set_ylabel("Link margin (dB)")
    axis.grid(True, linewidth=0.6, alpha=0.35)
    axis.legend()
    axis.margins(x=0)

    if Y_LIMITS_DB is not None:
        axis.set_ylim(*Y_LIMITS_DB)

    figure.tight_layout()
    save_figure(figure, OUTPUT_DIRECTORY / output_filename_stem)


def print_summary(
    loaded_data: dict[str, dict[str, pd.DataFrame]],
    mission_start: pd.Timestamp,
) -> None:
    """Print basic statistics for every imported link."""
    print("\nLoaded link-budget workbooks")
    print("=" * 90)
    print(f"Mission start: {mission_start}")
    print()

    for group_name, group_data in loaded_data.items():
        print(group_name.replace("_", " ").title())

        for station_key, dataframe in group_data.items():
            margin = dataframe["margin_db"].to_numpy(dtype=float)
            nonnegative_fraction = 100.0 * np.mean(margin >= 0.0)

            print(
                f"  {DSN_LABELS[station_key]:<22}"
                f" rows={len(dataframe):>7d}"
                f"  min={np.min(margin):>9.3f} dB"
                f"  max={np.max(margin):>9.3f} dB"
                f"  margin >= 0={nonnegative_fraction:>7.2f}%"
            )

        print()


def main() -> None:
    """Load all 12 workbooks and produce four margin plots."""
    if PLOT_EVERY_NTH_ROW < 1:
        raise ValueError("PLOT_EVERY_NTH_ROW must be at least 1.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    loaded_data: dict[str, dict[str, pd.DataFrame]] = {}

    for group_name, station_files in build_file_groups().items():
        loaded_data[group_name] = {
            station_key: read_link_budget_workbook(workbook_path)
            for station_key, workbook_path in station_files.items()
        }

    mission_start = determine_mission_start(loaded_data)

    plot_definitions = [
        (
            "uplink_lga1",
            "Mothership LGA 1 uplink margin",
            "uplink_lga1_margin_all_dsn",
        ),
        (
            "downlink_lga1",
            "Mothership LGA 1 downlink margin",
            "downlink_lga1_margin_all_dsn",
        ),
        (
            "uplink_lga2",
            "Mothership LGA 2 uplink margin",
            "uplink_lga2_margin_all_dsn",
        ),
        (
            "downlink_lga2",
            "Mothership LGA 2 downlink margin",
            "downlink_lga2_margin_all_dsn",
        ),
    ]

    for group_name, title, output_stem in plot_definitions:
        plot_margin_group(
            group_data=loaded_data[group_name],
            title=title,
            output_filename_stem=output_stem,
            mission_start=mission_start,
        )

    print_summary(loaded_data, mission_start)
    print(f"Figures saved to:\n{OUTPUT_DIRECTORY.resolve()}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()