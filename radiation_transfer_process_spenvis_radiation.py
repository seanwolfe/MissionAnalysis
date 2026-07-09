#!/usr/bin/env python3
"""Process paired SPENVIS NIEL/DDD and SHIELDOSE-2/TID exports.

The script accepts two SPENVIS text files:

1. a NIEL export containing displacement-damage summaries and shielded spectra;
2. a SHIELDOSE-2 export containing total ionizing dose versus Al shielding.

It writes one organized output folder containing metadata, clean CSV files, and
publication-style figures in PNG, PDF, and SVG formats.

PyCharm use
-----------
1. Set NIEL_INPUT_FILE and TID_INPUT_FILE in USER CONFIGURATION.
2. Optionally adjust the SEP assumptions and selected shielding depths.
3. Run the script.

Command-line use
----------------
python process_spenvis_radiation.py radiation_niel.txt radiation_tid.txt
python process_spenvis_radiation.py radiation_niel.txt radiation_tid.txt \
    --output-dir radiation_outputs
python process_spenvis_radiation.py radiation_niel.txt radiation_tid.txt --show

Output structure
----------------
<output>/
    README.txt
    metadata/
        niel_metadata.csv
        tid_metadata.csv
        combined_metadata.csv
        validation_report.csv
    csv/
        niel_summary.csv
        tid_summary.csv
        combined_radiation_summary.csv
        selected_shielding_summary.csv
        shielded_solar_proton_spectrum.csv          (when present)
        shielded_trapped_proton_spectrum.csv        (when present)
    figures/
        tid_vs_shielding.{png,pdf,svg}
        ddd_vs_shielding.{png,pdf,svg}
        equivalent_10MeV_proton_fluence_vs_shielding.{png,pdf,svg}
        tid_selected_shielding.{png,pdf,svg}
        ddd_selected_shielding.{png,pdf,svg}
        shielded_solar_proton_spectra.{png,pdf,svg} (when available)

Interpretation note
-------------------
The current SPENVIS ESP-PSYCHIC setup uses a 0.5-year solar-particle prediction
period. Therefore, the absolute solar-proton TID and DDD results are a common
0.5-year bounding environment, not a dose prediction specific to the uploaded
30-, 60-, 90-, or 120-day trajectory duration.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIGURATION — edit these values when running directly in PyCharm
# =============================================================================
NIEL_INPUT_FILE = Path(r"radiation_niel_T3B.txt")
TID_INPUT_FILE = Path(r"radiation_id_T3B.txt")

# None creates a project-based folder beside the NIEL input file.
OUTPUT_DIRECTORY: Path | None = None

# The SHIELDOSE-2 export does not retain the upstream SEP assumptions. These
# values should match the SPENVIS solar-particle calculation used before both
# the SHIELDOSE-2 and NIEL calculations.
SEP_MODEL = "ESP-PSYCHIC total fluence"
SEP_PREDICTION_PERIOD_YEARS = 0.5
SEP_SOLAR_MAXIMUM_YEARS = 0.5
SEP_SOLAR_MINIMUM_YEARS = 0.0
SEP_CONFIDENCE_PERCENT = 95.0
MAGNETIC_SHIELDING = "off"

ALUMINIUM_DENSITY_G_CM3 = 2.70

# Compact design-comparison table and bar plots. Values between SPENVIS grid
# points are interpolated in log-log space for positive quantities.
SELECTED_SHIELDING_DEPTHS_MM = (0.5, 1.0, 1.5, 2.0, 2.5, 5.0)

# Full shielded spectra are useful for traceability and the optional spectrum
# figure. They can make the CSV folder larger.
EXPORT_FULL_SPECTRA = True
MAKE_SPECTRUM_FIGURE = True
SPECTRUM_SHIELDING_MM = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)

# Keep the relative-degradation table for traceability, but do not interpret it
# as device degradation unless the SPENVIS damage factor is component-specific.
INCLUDE_RELATIVE_DEGRADATION = True

SAVE_FORMATS = ("png", "svg")
DPI = 400
SHOW_FIGURES = True
INCLUDE_TITLES = False
USE_LOG_X = True

# Optional vertical markers, for example:
# SHIELDING_MARKERS_MM = {"Mothership": 1.0, "Stowed microsatellite": 2.0}
SHIELDING_MARKERS_MM: dict[str, float] = {}
# =============================================================================


FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$"
)


@dataclass
class NielBlock:
    plot_type: str
    plot_header: str
    legends: list[str]
    x_name: str
    x_unit: str
    y_name: str
    y_unit: str
    rows: list[list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert paired SPENVIS NIEL and SHIELDOSE-2 text exports into "
            "one organized CSV-and-figure output folder."
        )
    )
    parser.add_argument(
        "niel_file",
        nargs="?",
        type=Path,
        help="SPENVIS NIEL/DDD text export. Overrides NIEL_INPUT_FILE.",
    )
    parser.add_argument(
        "tid_file",
        nargs="?",
        type=Path,
        help="SPENVIS SHIELDOSE-2/TID text export. Overrides TID_INPUT_FILE.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Combined output directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    parser.add_argument(
        "--no-spectrum",
        action="store_true",
        help="Do not create the shielded solar-proton spectrum figure.",
    )
    return parser.parse_args()


def parse_spenvis_csv_line(line: str) -> list[str]:
    """Parse a SPENVIS comma-separated record using single-quote quoting."""
    return [
        field.strip()
        for field in next(csv.reader([line], skipinitialspace=True, quotechar="'"))
    ]


def quoted_text(line: str) -> str | None:
    try:
        fields = parse_spenvis_csv_line(line)
    except (csv.Error, StopIteration):
        return None
    return fields[0].strip().strip("'").strip() if fields else None


def clean_key(text: str) -> str:
    key = text.strip().lower()
    key = key.replace("brems-", "brems")
    key = key.replace("nr.", "number")
    key = key.replace("10.0", "10_mev")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def record_value(lines: Iterable[str], name: str) -> str | None:
    prefix = f"'{name}'"
    for line in lines:
        if not line.lstrip().startswith(prefix):
            continue
        fields = parse_spenvis_csv_line(line)
        if len(fields) >= 3:
            return fields[2].strip().strip("'").strip()
    return None


def record_value_and_unit(lines: Iterable[str], name: str) -> tuple[str | None, str | None]:
    prefix = f"'{name}'"
    for line in lines:
        if not line.lstrip().startswith(prefix):
            continue
        fields = parse_spenvis_csv_line(line)
        value = fields[2].strip().strip("'").strip() if len(fields) >= 3 else None
        unit = fields[3].strip().strip("'").strip() if len(fields) >= 4 else None
        return value, unit
    return None, None


def find_annotation(lines: Iterable[str], prefix: str) -> str | None:
    for line in lines:
        text = quoted_text(line)
        if text and text.startswith(prefix):
            return text[len(prefix):].strip()
    return None


def parse_numeric_row(line: str) -> list[float] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("'") or stripped.startswith("*"):
        return None

    fields = [field.strip() for field in stripped.split(",")]
    if not fields or any(not field for field in fields):
        return None
    if any(FLOAT_PATTERN.fullmatch(field) is None for field in fields):
        return None
    return [float(field) for field in fields]


def ensure_input_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} input file does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} input path is not a file: {resolved}")
    return resolved


# -----------------------------------------------------------------------------
# NIEL parsing
# -----------------------------------------------------------------------------
def split_niel_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in text.splitlines():
        current.append(line.rstrip())
        terminator = line.strip().strip("'")
        if terminator in {"End of Block", "End of File"}:
            blocks.append(current)
            current = []

    if current and any(item.strip() for item in current):
        blocks.append(current)
    return blocks


def parse_niel_block(lines: list[str]) -> NielBlock | None:
    plot_type = record_value(lines, "PLT_TYP") or ""
    plot_header = record_value(lines, "PLT_HDR") or ""

    legends: list[str] = []
    for line in lines:
        if line.lstrip().startswith("'PLT_LEG'"):
            fields = parse_spenvis_csv_line(line)
            legends = [field.strip().strip("'").strip() for field in fields[2:]]
            break

    data_start: int | None = None
    x_fields: list[str] = []
    y_fields: list[str] = []

    for index in range(len(lines) - 1):
        first = lines[index].lstrip()
        second = lines[index + 1].lstrip()
        if not (first.startswith("'Energy'") or first.startswith("'Thick'")):
            continue
        if not second.startswith("'Dose'"):
            continue
        x_fields = parse_spenvis_csv_line(lines[index])
        y_fields = parse_spenvis_csv_line(lines[index + 1])
        data_start = index + 2
        break

    if data_start is None:
        return None

    x_name = x_fields[3].strip().strip("'") if len(x_fields) >= 4 else x_fields[0]
    x_unit = x_fields[1].strip().strip("'") if len(x_fields) >= 2 else ""
    y_name = y_fields[3].strip().strip("'") if len(y_fields) >= 4 else y_fields[0]
    y_unit = y_fields[1].strip().strip("'") if len(y_fields) >= 2 else ""

    rows: list[list[float]] = []
    for line in lines[data_start:]:
        if line.strip().strip("'") in {"End of Block", "End of File"}:
            break
        numeric = parse_numeric_row(line)
        if numeric is not None:
            rows.append(numeric)

    if not rows:
        return None

    expected_columns = 1 + len(legends)
    bad_lengths = sorted({len(row) for row in rows if len(row) != expected_columns})
    if legends and bad_lengths:
        raise ValueError(
            f"Unexpected column count in NIEL block '{plot_header or y_name}'. "
            f"Expected {expected_columns}; found {bad_lengths}."
        )

    return NielBlock(
        plot_type=plot_type,
        plot_header=plot_header,
        legends=legends,
        x_name=x_name,
        x_unit=x_unit,
        y_name=y_name,
        y_unit=y_unit,
        rows=rows,
    )


def parse_niel_blocks(text: str) -> list[NielBlock]:
    blocks = [
        block
        for raw in split_niel_blocks(text)
        if (block := parse_niel_block(raw)) is not None
    ]
    if not blocks:
        raise ValueError("No SPENVIS NIEL numeric blocks were found.")
    return blocks


def niel_summary_kind(block: NielBlock) -> tuple[str, str] | None:
    description = block.y_name.lower()
    if "displacement damage dose" in description:
        return "ddd", "MeV_g"
    if "damage equivalent" in description and "proton fluence" in description:
        return "equivalent_10MeV_proton_fluence", "cm2"
    if "relative degradation" in description:
        return "relative_degradation", "dimensionless"
    return None


def niel_legend_column(legend: str, quantity: str, unit_suffix: str) -> str:
    cleaned = clean_key(legend)
    source = {
        "total": "total",
        "trapped_protons": "trapped_protons",
        "solar_protons": "solar_protons",
    }.get(cleaned, cleaned)
    return f"{quantity}_{source}_{unit_suffix}"


def build_niel_summary(blocks: Sequence[NielBlock]) -> pd.DataFrame:
    summary_blocks: list[tuple[NielBlock, str, str]] = []
    for block in blocks:
        if block.plot_type.upper() != "SUMMARY":
            continue
        kind = niel_summary_kind(block)
        if kind is not None:
            summary_blocks.append((block, kind[0], kind[1]))

    required = {"ddd", "equivalent_10MeV_proton_fluence"}
    found = {kind for _, kind, _ in summary_blocks}
    missing = required - found
    if missing:
        raise ValueError(
            "Missing required NIEL summary table(s): " + ", ".join(sorted(missing))
        )

    if not INCLUDE_RELATIVE_DEGRADATION:
        summary_blocks = [item for item in summary_blocks if item[1] != "relative_degradation"]

    merged: dict[float, dict[str, float]] = {}
    column_order: list[str] = []

    for block, quantity, unit_suffix in summary_blocks:
        columns = [
            niel_legend_column(legend, quantity, unit_suffix)
            for legend in block.legends
        ]
        for column in columns:
            if column not in column_order:
                column_order.append(column)

        for row in block.rows:
            depth = float(row[0])
            destination = merged.setdefault(depth, {})
            for column, value in zip(columns, row[1:], strict=True):
                destination[column] = float(value)

    rows: list[dict[str, float]] = []
    for depth in sorted(merged):
        row: dict[str, float] = {"shielding_thickness_mm": depth}
        row.update(merged[depth])
        missing_columns = [column for column in column_order if column not in row]
        if missing_columns:
            raise ValueError(
                f"NIEL shielding depth {depth:g} mm is missing columns: "
                + ", ".join(missing_columns)
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(
        1,
        "shielding_thickness_mils",
        df["shielding_thickness_mm"] / 0.0254,
    )
    df.insert(
        2,
        "shielding_areal_density_g_cm2",
        (df["shielding_thickness_mm"] / 10.0) * ALUMINIUM_DENSITY_G_CM3,
    )
    return df


def niel_spectrum_headers(block: NielBlock) -> list[str]:
    headers = ["energy_MeV"]
    for legend in block.legends:
        normalized = legend.strip().lower()
        if normalized == "unshielded":
            headers.append("unshielded_integral_fluence_cm2")
            continue
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm", normalized)
        if match:
            encoded = match.group(1).replace(".", "p")
            headers.append(f"shield_{encoded}_mm_integral_fluence_cm2")
        else:
            headers.append(f"{clean_key(legend)}_integral_fluence_cm2")
    return headers


def find_niel_spectrum_block(
    blocks: Sequence[NielBlock],
    particle: str,
) -> NielBlock | None:
    target = f"shielded {particle} proton spectrum"
    for block in blocks:
        if block.plot_header.strip().lower() == target:
            return block
    return None


def extract_niel_metadata(text: str, input_file: Path) -> pd.DataFrame:
    lines = text.splitlines()
    values: dict[str, str] = {
        "source_file": input_file.name,
        "conversion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    for line in lines:
        first = quoted_text(line)
        if first and first.startswith("SPENVIS "):
            values["spenvis_export_header"] = first
            break

    record_map = {
        "PRJ_DEF": "project_definition",
        "PRJ_HDR": "project_title",
        "MOD_ABB": "module_abbreviation",
        "MIS_DUR": "trajectory_duration_days",
        "ORB_HDR": "trajectory_name",
        "ORB_TYP": "trajectory_type",
        "ORB_MJR": "orbit_time_reference",
        "ORB_GDR": "coordinate_reference_citation",
    }
    for record, key in record_map.items():
        value = record_value(lines, record)
        if value is not None:
            values[key] = value

    annotation_prefixes = {
        "Mission start:": "mission_start",
        "Mission end:": "mission_end",
        "Nr. of segments:": "number_of_segments",
        "Segment  1:": "segment_1_name",
        "Orbit start:": "orbit_start",
        "Orbit end:": "orbit_end",
        "Trapped proton model:": "trapped_proton_model",
        "Solar particle model:": "solar_particle_model",
        "Mission duration:": "sep_prediction_period_description",
        "Confidence level:": "sep_confidence_level",
        "Magnetic shielding:": "magnetic_shielding",
        "Relative degradation per unit NIEL:": "relative_degradation_factor",
        "Damage equivalent proton energy:": "reference_proton_energy",
        "NIEL damage curve:": "niel_damage_curve",
    }

    for line in lines:
        first = quoted_text(line)
        if not first:
            continue
        for prefix, key in annotation_prefixes.items():
            if first.startswith(prefix) and key not in values:
                values[key] = first[len(prefix):].strip()
                break
        if "years in solar maximum" in first and "sep_years_solar_maximum" not in values:
            values["sep_years_solar_maximum"] = first.split("years", 1)[0].strip()
        elif "years in solar minimum" in first and "sep_years_solar_minimum" not in values:
            values["sep_years_solar_minimum"] = first.split("years", 1)[0].strip()

    values["shielding_geometry"] = "centre of spherical aluminium shield"
    values["shielding_depth_unit"] = "mm Al-equivalent"
    return pd.DataFrame(values.items(), columns=["parameter", "value"])


# -----------------------------------------------------------------------------
# TID parsing
# -----------------------------------------------------------------------------
def parse_tid_table(text: str) -> tuple[list[str], list[list[float]], dict[str, str]]:
    lines = text.splitlines()

    legends: list[str] = []
    for line in lines:
        if line.lstrip().startswith("'PLT_LEG'"):
            fields = parse_spenvis_csv_line(line)
            legends = [field.strip().strip("'") for field in fields[2:]]
            break
    if not legends:
        raise ValueError("Could not find the SHIELDOSE-2 PLT_LEG record.")

    data_start: int | None = None
    table_info = {"shielding_unit": "", "dose_unit": "", "dose_description": ""}

    for index in range(len(lines) - 1):
        first = lines[index].lstrip()
        second = lines[index + 1].lstrip()
        if first.startswith("'Thick'") and second.startswith("'Dose'"):
            x_fields = parse_spenvis_csv_line(lines[index])
            y_fields = parse_spenvis_csv_line(lines[index + 1])
            table_info["shielding_unit"] = (
                x_fields[1].strip().strip("'") if len(x_fields) >= 2 else ""
            )
            table_info["dose_unit"] = (
                y_fields[1].strip().strip("'") if len(y_fields) >= 2 else ""
            )
            table_info["dose_description"] = (
                y_fields[3].strip().strip("'") if len(y_fields) >= 4 else ""
            )
            data_start = index + 2
            break

    if data_start is None:
        raise ValueError("Could not find the SHIELDOSE-2 thickness/dose table.")

    rows: list[list[float]] = []
    for line in lines[data_start:]:
        if line.strip().strip("'") in {"End of File", "End of Block"}:
            break
        numeric = parse_numeric_row(line)
        if numeric is not None:
            rows.append(numeric)
    if not rows:
        raise ValueError("No SHIELDOSE-2 numeric rows were found.")

    expected_columns = 1 + len(legends)
    bad_lengths = sorted({len(row) for row in rows if len(row) != expected_columns})
    if bad_lengths:
        raise ValueError(
            f"Expected {expected_columns} SHIELDOSE-2 numeric columns; found {bad_lengths}."
        )

    return legends, rows, table_info


def build_tid_summary(legends: Sequence[str], rows: Sequence[Sequence[float]]) -> pd.DataFrame:
    source_keys = [clean_key(name) for name in legends]
    df = pd.DataFrame(rows, columns=["shielding_thickness_mm", *source_keys])
    df = df.sort_values("shielding_thickness_mm").reset_index(drop=True)

    if (df["shielding_thickness_mm"] <= 0).any():
        raise ValueError("TID shielding thicknesses must be positive.")
    if df["shielding_thickness_mm"].duplicated().any():
        raise ValueError("Duplicate TID shielding thicknesses were found.")

    df.insert(
        1,
        "shielding_thickness_mils",
        df["shielding_thickness_mm"] / 0.0254,
    )
    df.insert(
        2,
        "shielding_areal_density_g_cm2",
        (df["shielding_thickness_mm"] / 10.0) * ALUMINIUM_DENSITY_G_CM3,
    )

    df = df.rename(columns={key: f"tid_{key}_rad_si" for key in source_keys})
    dose_columns = [column for column in df.columns if column.endswith("_rad_si")]
    for column in dose_columns:
        df[column.replace("_rad_si", "_krad_si")] = df[column] / 1000.0

    if "tid_total_rad_si" not in df.columns:
        raise ValueError("The SHIELDOSE-2 table does not contain a Total dose column.")

    component_columns = [column for column in dose_columns if column != "tid_total_rad_si"]
    if component_columns:
        df["dose_component_closure_error_rad_si"] = (
            df["tid_total_rad_si"] - df[component_columns].sum(axis=1)
        )

    preferred = [
        "shielding_thickness_mm",
        "shielding_thickness_mils",
        "shielding_areal_density_g_cm2",
        "tid_total_rad_si",
        "tid_total_krad_si",
    ]
    for source in ("electrons", "bremsstrahlung", "trapped_protons", "solar_protons"):
        for suffix in ("rad_si", "krad_si"):
            column = f"tid_{source}_{suffix}"
            if column in df.columns:
                preferred.append(column)
    if "dose_component_closure_error_rad_si" in df.columns:
        preferred.append("dose_component_closure_error_rad_si")
    remaining = [column for column in df.columns if column not in preferred]
    return df[[*preferred, *remaining]]


def extract_tid_metadata(
    text: str,
    input_file: Path,
    table_info: dict[str, str],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    lines = text.splitlines()
    duration_value, duration_unit = record_value_and_unit(lines, "MIS_DUR")

    values: list[tuple[str, str]] = [
        ("source_file", input_file.name),
        ("conversion_timestamp_utc", datetime.now(timezone.utc).isoformat()),
    ]

    for line in lines:
        first = quoted_text(line)
        if first and first.startswith("SPENVIS "):
            values.append(("spenvis_export_header", first))
            break

    for record, key in {
        "PRJ_DEF": "project_definition",
        "PRJ_HDR": "project_title",
        "MOD_ABB": "module_abbreviation",
        "PLT_TYP": "plot_type",
        "PLT_HDR": "plot_header",
    }.items():
        value = record_value(lines, record)
        if value is not None:
            values.append((key, value))

    if duration_value is not None:
        values.append(("trajectory_duration_days", duration_value))
    if duration_unit is not None:
        values.append(("trajectory_duration_unit", duration_unit))

    for prefix, key in (
        ("Mission start:", "mission_start"),
        ("Mission end:", "mission_end"),
        ("Duration:", "mission_duration_annotation"),
        ("Nr. of segments:", "number_of_segments"),
    ):
        value = find_annotation(lines, prefix)
        if value is not None:
            values.append((key, value))

    values.extend(
        [
            ("dose_model", "SHIELDOSE-2"),
            ("shielding_configuration", "centre of spherical aluminium shield"),
            ("target_material", "Silicon"),
            ("dose_unit_in_source", table_info.get("dose_unit", "rad")),
            ("dose_description", table_info.get("dose_description", "Dose in Si")),
            ("aluminium_density_g_cm3", f"{ALUMINIUM_DENSITY_G_CM3:g}"),
            ("sep_model_user_assumption", SEP_MODEL),
            (
                "sep_prediction_period_years_user_assumption",
                f"{SEP_PREDICTION_PERIOD_YEARS:g}",
            ),
            (
                "sep_solar_maximum_years_user_assumption",
                f"{SEP_SOLAR_MAXIMUM_YEARS:g}",
            ),
            (
                "sep_solar_minimum_years_user_assumption",
                f"{SEP_SOLAR_MINIMUM_YEARS:g}",
            ),
            (
                "sep_confidence_percent_user_assumption",
                f"{SEP_CONFIDENCE_PERCENT:g}",
            ),
            ("magnetic_shielding_user_assumption", MAGNETIC_SHIELDING),
            (
                "shielding_range_mm",
                f"{summary['shielding_thickness_mm'].min():g} to "
                f"{summary['shielding_thickness_mm'].max():g}",
            ),
            (
                "interpretation_note",
                "Absolute solar-proton dose reflects the configured SEP prediction "
                "period rather than the uploaded trajectory duration.",
            ),
        ]
    )
    return pd.DataFrame(values, columns=["parameter", "value"])


# -----------------------------------------------------------------------------
# Combined outputs and validation
# -----------------------------------------------------------------------------
def metadata_dict(metadata: pd.DataFrame) -> dict[str, str]:
    return dict(zip(metadata["parameter"].astype(str), metadata["value"].astype(str)))


def normalize_numeric_text(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", value)
    return float(match.group(0)) if match else None


def build_validation_report(
    niel_metadata: pd.DataFrame,
    tid_metadata: pd.DataFrame,
    niel_summary: pd.DataFrame,
    tid_summary: pd.DataFrame,
) -> pd.DataFrame:
    niel = metadata_dict(niel_metadata)
    tid = metadata_dict(tid_metadata)
    checks: list[dict[str, str]] = []

    def add_check(name: str, niel_value: object, tid_value: object, status: str, note: str = "") -> None:
        checks.append(
            {
                "check": name,
                "niel_value": "" if niel_value is None else str(niel_value),
                "tid_value": "" if tid_value is None else str(tid_value),
                "status": status,
                "note": note,
            }
        )

    for key in ("project_definition", "project_title", "mission_start", "mission_end"):
        nv = niel.get(key)
        tv = tid.get(key)
        if nv is None or tv is None:
            add_check(key, nv, tv, "not_checked", "Value absent from one export.")
        else:
            add_check(key, nv, tv, "pass" if nv == tv else "warning")

    nd = normalize_numeric_text(niel.get("trajectory_duration_days"))
    td = normalize_numeric_text(tid.get("trajectory_duration_days"))
    if nd is None or td is None:
        add_check("trajectory_duration_days", nd, td, "not_checked")
    else:
        status = "pass" if math.isclose(nd, td, rel_tol=0.0, abs_tol=1.0e-6) else "warning"
        add_check("trajectory_duration_days", nd, td, status)

    niel_depths = niel_summary["shielding_thickness_mm"].to_numpy(dtype=float)
    tid_depths = tid_summary["shielding_thickness_mm"].to_numpy(dtype=float)
    same_grid = (
        len(niel_depths) == len(tid_depths)
        and np.allclose(niel_depths, tid_depths, rtol=0.0, atol=1.0e-12)
    )
    add_check(
        "shielding_grid",
        f"{len(niel_depths)} rows, {niel_depths.min():g}-{niel_depths.max():g} mm",
        f"{len(tid_depths)} rows, {tid_depths.min():g}-{tid_depths.max():g} mm",
        "pass" if same_grid else "warning",
        "An outer merge is used when grids differ.",
    )

    return pd.DataFrame(checks)


def build_combined_metadata(
    niel_metadata: pd.DataFrame,
    tid_metadata: pd.DataFrame,
) -> pd.DataFrame:
    niel = metadata_dict(niel_metadata)
    tid = metadata_dict(tid_metadata)
    keys = sorted(set(niel) | set(tid))
    rows: list[dict[str, str]] = []
    for key in keys:
        nv = niel.get(key, "")
        tv = tid.get(key, "")
        if nv and tv:
            status = "match" if nv == tv else "different"
        elif nv:
            status = "niel_only"
        else:
            status = "tid_only"
        rows.append(
            {
                "parameter": key,
                "niel_value": nv,
                "tid_value": tv,
                "comparison": status,
            }
        )
    return pd.DataFrame(rows)


def build_combined_summary(
    niel_summary: pd.DataFrame,
    tid_summary: pd.DataFrame,
) -> pd.DataFrame:
    common_geometry_columns = [
        "shielding_thickness_mm",
        "shielding_thickness_mils",
        "shielding_areal_density_g_cm2",
    ]
    niel_extra = [column for column in niel_summary.columns if column not in common_geometry_columns]
    tid_extra = [column for column in tid_summary.columns if column not in common_geometry_columns]

    combined = pd.merge(
        tid_summary[common_geometry_columns + tid_extra],
        niel_summary[common_geometry_columns + niel_extra],
        on="shielding_thickness_mm",
        how="outer",
        suffixes=("_tid", "_niel"),
        validate="one_to_one",
    )

    # Use one canonical set of geometry columns and remove duplicated merge columns.
    for column in ("shielding_thickness_mils", "shielding_areal_density_g_cm2"):
        left = f"{column}_tid"
        right = f"{column}_niel"
        if left in combined.columns and right in combined.columns:
            combined[column] = combined[left].combine_first(combined[right])
            combined = combined.drop(columns=[left, right])

    first = [
        "shielding_thickness_mm",
        "shielding_thickness_mils",
        "shielding_areal_density_g_cm2",
    ]
    remaining = [column for column in combined.columns if column not in first]
    return combined[[*first, *remaining]].sort_values("shielding_thickness_mm").reset_index(drop=True)


def log_log_interpolate(x: np.ndarray, y: np.ndarray, target: float) -> tuple[float, bool]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        raise ValueError("No valid values are available for interpolation.")
    if target < float(np.min(x)) or target > float(np.max(x)):
        raise ValueError(
            f"Selected shielding depth {target:g} mm is outside the available "
            f"range {np.min(x):g}-{np.max(x):g} mm."
        )
    exact = np.isclose(x, target, rtol=0.0, atol=1.0e-12)
    if exact.any():
        return float(y[np.flatnonzero(exact)[0]]), False
    if np.any(y <= 0):
        return float(np.interp(target, x, y)), True
    value = 10.0 ** np.interp(math.log10(target), np.log10(x), np.log10(y))
    return float(value), True


def build_selected_summary(
    niel_summary: pd.DataFrame,
    tid_summary: pd.DataFrame,
) -> pd.DataFrame:
    tid_x = tid_summary["shielding_thickness_mm"].to_numpy(dtype=float)
    niel_x = niel_summary["shielding_thickness_mm"].to_numpy(dtype=float)

    tid_y = tid_summary["tid_total_krad_si"].to_numpy(dtype=float)
    ddd_y = niel_summary["ddd_total_MeV_g"].to_numpy(dtype=float)
    eq_y = niel_summary[
        "equivalent_10MeV_proton_fluence_total_cm2"
    ].to_numpy(dtype=float)

    rows: list[dict[str, float | bool]] = []
    for depth in SELECTED_SHIELDING_DEPTHS_MM:
        tid_value, tid_interp = log_log_interpolate(tid_x, tid_y, float(depth))
        ddd_value, ddd_interp = log_log_interpolate(niel_x, ddd_y, float(depth))
        eq_value, eq_interp = log_log_interpolate(niel_x, eq_y, float(depth))
        rows.append(
            {
                "shielding_thickness_mm": float(depth),
                "shielding_thickness_mils": float(depth) / 0.0254,
                "shielding_areal_density_g_cm2":
                    (float(depth) / 10.0) * ALUMINIUM_DENSITY_G_CM3,
                "tid_total_krad_si": tid_value,
                "ddd_total_MeV_g": ddd_value,
                "equivalent_10MeV_proton_fluence_total_cm2": eq_value,
                "tid_interpolated": tid_interp,
                "ddd_interpolated": ddd_interp,
                "equivalent_fluence_interpolated": eq_interp,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    for extension in SAVE_FORMATS:
        output_path = output_base.with_suffix(f".{extension}")
        kwargs = {"dpi": DPI} if extension.lower() == "png" else {}
        fig.savefig(output_path, **kwargs)
        print(f"Saved: {output_path}")


def add_shielding_markers(ax: plt.Axes) -> None:
    for label, depth_mm in SHIELDING_MARKERS_MM.items():
        if depth_mm <= 0:
            raise ValueError(f"Shielding marker '{label}' must be positive.")
        ax.axvline(depth_mm, linestyle="--", linewidth=1.0, label=label)


def format_shielding_axis(ax: plt.Axes, x: np.ndarray) -> None:
    if USE_LOG_X:
        ax.set_xscale("log")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_xlabel("Aluminium-equivalent shielding depth (mm)")
    ax.grid(True, which="major", linestyle=":", linewidth=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.6)


def plot_log_curve(
    x: np.ndarray,
    y: np.ndarray,
    ylabel: str,
    label: str,
    marker: str,
    output_base: Path,
    title: str,
) -> None:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if not np.any(valid):
        raise ValueError(f"No positive data available for figure: {output_base.name}")

    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    ax.plot(x[valid], y[valid], marker=marker, markerfacecolor="white", label=label)
    ax.set_yscale("log")
    format_shielding_axis(ax, x[valid])
    ax.set_ylabel(ylabel)
    add_shielding_markers(ax)
    if INCLUDE_TITLES:
        ax.set_title(title)
    if SHIELDING_MARKERS_MM:
        ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_selected_bars(
    selected: pd.DataFrame,
    value_column: str,
    ylabel: str,
    output_base: Path,
    title: str,
) -> None:
    labels = [f"{value:g}" for value in selected["shielding_thickness_mm"]]
    values = selected[value_column].to_numpy(dtype=float)
    valid = np.isfinite(values) & (values > 0)
    labels = [label for label, keep in zip(labels, valid, strict=True) if keep]
    values = values[valid]

    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    bars = ax.bar(labels, values)
    ax.set_xlabel("Aluminium-equivalent shielding depth (mm)")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.6)
    ax.grid(True, axis="y", which="minor", linestyle=":", linewidth=0.35, alpha=0.6)

    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.3g}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if INCLUDE_TITLES:
        ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, output_base)


def spectrum_column_for_depth(columns: Iterable[str], depth_mm: float) -> str:
    if depth_mm == 0:
        expected = "unshielded_integral_fluence_cm2"
    else:
        expected = (
            f"shield_{depth_mm:.3f}".replace(".", "p")
            + "_mm_integral_fluence_cm2"
        )
    if expected in columns:
        return expected

    available: list[float] = []
    pattern = re.compile(r"^shield_(\d+)p(\d+)_mm_integral_fluence_cm2$")
    for column in columns:
        match = pattern.match(column)
        if match:
            available.append(float(f"{match.group(1)}.{match.group(2)}"))
    raise ValueError(
        f"No spectrum column exists for {depth_mm:g} mm. Available: {sorted(available)}"
    )


def plot_solar_spectra(spectrum: pd.DataFrame, output_base: Path) -> None:
    if "energy_MeV" not in spectrum.columns:
        raise ValueError("The solar-proton spectrum CSV does not contain energy_MeV.")

    energy = pd.to_numeric(spectrum["energy_MeV"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    markers = ("o", "s", "^", "D", "v", "P")

    for index, depth in enumerate(SPECTRUM_SHIELDING_MM):
        column = spectrum_column_for_depth(spectrum.columns, depth)
        fluence = pd.to_numeric(spectrum[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(energy) & np.isfinite(fluence) & (energy > 0) & (fluence > 0)
        label = "Unshielded" if depth == 0 else f"{depth:g} mm Al"
        ax.plot(
            energy[valid],
            fluence[valid],
            marker=markers[index % len(markers)],
            markevery=max(1, int(np.count_nonzero(valid) / 12)),
            markerfacecolor="white",
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Proton energy (MeV)")
    ax.set_ylabel(r"Integral solar-proton fluence (cm$^{-2}$)")
    ax.grid(True, which="major", linestyle=":", linewidth=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.6)
    ax.legend(frameon=False, ncol=2)
    if INCLUDE_TITLES:
        ax.set_title("Shielded solar-proton integral spectra")
    fig.tight_layout()
    save_figure(fig, output_base)


# -----------------------------------------------------------------------------
# File writing and main workflow
# -----------------------------------------------------------------------------
def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("_") or "spenvis"


def default_output_directory(
    niel_file: Path,
    niel_metadata: pd.DataFrame,
) -> Path:
    metadata = metadata_dict(niel_metadata)
    project = metadata.get("project_definition") or metadata.get("project_title")
    if project:
        name = f"{sanitize_filename(project)}_radiation_outputs"
    else:
        name = f"{niel_file.stem}_combined_radiation_outputs"
    return niel_file.parent / name


def write_readme(
    path: Path,
    niel_file: Path,
    tid_file: Path,
    output_dir: Path,
) -> None:
    text = f"""SPENVIS combined radiation outputs
==================================

Source files
------------
NIEL/DDD:       {niel_file.name}
SHIELDOSE/TID:  {tid_file.name}

Output directory
----------------
{output_dir}

Radiation assumptions used for interpretation
---------------------------------------------
SEP model:                  {SEP_MODEL}
SEP prediction period:      {SEP_PREDICTION_PERIOD_YEARS:g} years
Time in solar maximum:      {SEP_SOLAR_MAXIMUM_YEARS:g} years
Time in solar minimum:      {SEP_SOLAR_MINIMUM_YEARS:g} years
Confidence level:           {SEP_CONFIDENCE_PERCENT:g}%
Magnetic shielding:         {MAGNETIC_SHIELDING}
Shielding geometry:         centre of spherical aluminium shield
Target material:            silicon

Important interpretation
------------------------
The solar-proton TID and DDD values correspond to the configured 0.5-year
ESP-PSYCHIC prediction period. They are a common conservative bounding case,
not a dose prediction specific to the uploaded trajectory duration.

The NIEL relative-degradation columns use the SPENVIS input damage factor. Do
not interpret those values as detector performance loss unless the damage
factor is component-specific.

Folder contents
---------------
metadata/  Source metadata, cross-file comparison, and validation checks.
csv/       Clean TID, DDD, equivalent-fluence, spectra, and merged summaries.
figures/   Publication-style PNG, PDF, and SVG figures.
"""
    path.write_text(text, encoding="utf-8")


def process(
    niel_file: Path,
    tid_file: Path,
    output_dir_override: Path | None,
    make_spectrum_figure: bool,
) -> Path:
    niel_file = ensure_input_file(niel_file, "NIEL")
    tid_file = ensure_input_file(tid_file, "TID")

    niel_text = niel_file.read_text(encoding="utf-8", errors="replace")
    tid_text = tid_file.read_text(encoding="utf-8", errors="replace")

    niel_blocks = parse_niel_blocks(niel_text)
    niel_summary = build_niel_summary(niel_blocks)
    niel_metadata = extract_niel_metadata(niel_text, niel_file)

    tid_legends, tid_rows, tid_info = parse_tid_table(tid_text)
    tid_summary = build_tid_summary(tid_legends, tid_rows)
    tid_metadata = extract_tid_metadata(tid_text, tid_file, tid_info, tid_summary)

    if output_dir_override is not None:
        output_dir = output_dir_override.expanduser().resolve()
    elif OUTPUT_DIRECTORY is not None:
        output_dir = OUTPUT_DIRECTORY.expanduser().resolve()
    else:
        output_dir = default_output_directory(niel_file, niel_metadata).resolve()

    metadata_dir = output_dir / "metadata"
    csv_dir = output_dir / "csv"
    figures_dir = output_dir / "figures"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    combined_metadata = build_combined_metadata(niel_metadata, tid_metadata)
    validation = build_validation_report(
        niel_metadata,
        tid_metadata,
        niel_summary,
        tid_summary,
    )
    combined_summary = build_combined_summary(niel_summary, tid_summary)
    selected_summary = build_selected_summary(niel_summary, tid_summary)

    niel_metadata.to_csv(metadata_dir / "niel_metadata.csv", index=False)
    tid_metadata.to_csv(metadata_dir / "tid_metadata.csv", index=False)
    combined_metadata.to_csv(metadata_dir / "combined_metadata.csv", index=False)
    validation.to_csv(metadata_dir / "validation_report.csv", index=False)

    niel_summary.to_csv(csv_dir / "niel_summary.csv", index=False, float_format="%.10g")
    tid_summary.to_csv(csv_dir / "tid_summary.csv", index=False, float_format="%.10g")
    combined_summary.to_csv(
        csv_dir / "combined_radiation_summary.csv",
        index=False,
        float_format="%.10g",
    )
    selected_summary.to_csv(
        csv_dir / "selected_shielding_summary.csv",
        index=False,
        float_format="%.10g",
    )

    solar_spectrum: pd.DataFrame | None = None
    if EXPORT_FULL_SPECTRA:
        for particle, filename in (
            ("solar", "shielded_solar_proton_spectrum.csv"),
            ("trapped", "shielded_trapped_proton_spectrum.csv"),
        ):
            block = find_niel_spectrum_block(niel_blocks, particle)
            if block is None:
                print(f"Warning: {particle}-proton spectrum block not found.", file=sys.stderr)
                continue
            spectrum = pd.DataFrame(block.rows, columns=niel_spectrum_headers(block))
            spectrum.to_csv(csv_dir / filename, index=False, float_format="%.10g")
            if particle == "solar":
                solar_spectrum = spectrum

    write_readme(output_dir / "README.txt", niel_file, tid_file, output_dir)

    configure_matplotlib()
    tid_x = tid_summary["shielding_thickness_mm"].to_numpy(dtype=float)
    niel_x = niel_summary["shielding_thickness_mm"].to_numpy(dtype=float)

    plot_log_curve(
        tid_x,
        tid_summary["tid_total_krad_si"].to_numpy(dtype=float),
        r"Total ionizing dose (krad(Si))",
        "Total TID",
        "o",
        figures_dir / "tid_vs_shielding",
        "Total ionizing dose versus shielding depth",
    )
    plot_log_curve(
        niel_x,
        niel_summary["ddd_total_MeV_g"].to_numpy(dtype=float),
        r"Displacement damage dose (MeV g$^{-1}$)",
        "Total DDD",
        "o",
        figures_dir / "ddd_vs_shielding",
        "Displacement damage dose versus shielding depth",
    )
    plot_log_curve(
        niel_x,
        niel_summary[
            "equivalent_10MeV_proton_fluence_total_cm2"
        ].to_numpy(dtype=float),
        r"10-MeV proton-equivalent fluence (cm$^{-2}$)",
        "10 MeV p equivalent",
        "s",
        figures_dir / "equivalent_10MeV_proton_fluence_vs_shielding",
        "Damage-equivalent proton fluence versus shielding depth",
    )
    plot_selected_bars(
        selected_summary,
        "tid_total_krad_si",
        r"Total ionizing dose (krad(Si))",
        figures_dir / "tid_selected_shielding",
        "TID at selected shielding depths",
    )
    plot_selected_bars(
        selected_summary,
        "ddd_total_MeV_g",
        r"Displacement damage dose (MeV g$^{-1}$)",
        figures_dir / "ddd_selected_shielding",
        "DDD at selected shielding depths",
    )
    if make_spectrum_figure and solar_spectrum is not None:
        plot_solar_spectra(
            solar_spectrum,
            figures_dir / "shielded_solar_proton_spectra",
        )

    print("\nCombined SPENVIS radiation workflow complete.")
    print(f"NIEL input: {niel_file}")
    print(f"TID input:  {tid_file}")
    print(f"Output:     {output_dir}")
    print(f"NIEL rows:  {len(niel_summary):,}")
    print(f"TID rows:   {len(tid_summary):,}")
    warnings = validation.loc[validation["status"] == "warning"]
    if not warnings.empty:
        print("\nValidation warnings:")
        print(warnings[["check", "niel_value", "tid_value"]].to_string(index=False))
    return output_dir


def main() -> int:
    args = parse_args()
    niel_file = args.niel_file if args.niel_file is not None else NIEL_INPUT_FILE
    tid_file = args.tid_file if args.tid_file is not None else TID_INPUT_FILE

    output_dir = process(
        niel_file=niel_file,
        tid_file=tid_file,
        output_dir_override=args.output_dir,
        make_spectrum_figure=MAKE_SPECTRUM_FIGURE and not args.no_spectrum,
    )

    if args.show or SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")

    print(f"\nCreated combined output folder: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
