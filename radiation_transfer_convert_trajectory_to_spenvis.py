#!/usr/bin/env python3
"""Convert one geocentric EME2000 trajectory CSV to SPENVIS Earth GEI format.

Input coordinates:
    Earth-centred EME2000/J2000 Cartesian position, km.

Output coordinates:
    Earth-centred GEI Cartesian position, km, referred to Earth's mean
    equator and dynamical equinox of B1950, as required by the SPENVIS
    trajectory-upload format.

The original trajectory epochs and sampling are preserved. Velocity and other
input columns are not written because SPENVIS requires only time and position.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# PyCharm-friendly defaults
# -----------------------------------------------------------------------------
# Set INPUT_CSV to your trajectory file and run the script directly, or provide
# the input path on the command line. A command-line path overrides INPUT_CSV.
INPUT_CSV = Path(r"best_dv_296749_TCM-1_plus_TCM-2_plus_TCM-3_plus_ACQ_trajectory.csv")

# Leave as None to create <input_stem>_spenvis_gei_b1950.txt beside the CSV.
OUTPUT_FILE: Path | None = None

# Leave as None to derive the title from the input filename.
SPENVIS_TITLE: str | None = None

# Number of digits after the decimal for position values in km.
POSITION_DECIMALS = 9

# Number of digits after the decimal for calendar seconds.
SECOND_DECIMALS = 6


REQUIRED_COLUMNS = (
    "utc",
    "x_geo_j2000_km",
    "y_geo_j2000_km",
    "z_geo_j2000_km",
)


# SPICE/NAIF B1950 definition. NAIF specifies the B1950-to-J2000
# rotation as R3(-z) R2(theta) R3(-zeta), with the IAU 1976 precession
# angles below. Its transpose maps J2000 position vectors into B1950.
ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)
B1950_Z_RAD = 1153.04066200330 * ARCSEC_TO_RAD
B1950_THETA_RAD = 1002.26108439117 * ARCSEC_TO_RAD
B1950_ZETA_RAD = 1152.84248596724 * ARCSEC_TO_RAD


def rotation_axis_2(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]],
        dtype=float,
    )


def rotation_axis_3(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


B1950_TO_J2000 = (
    rotation_axis_3(-B1950_Z_RAD)
    @ rotation_axis_2(B1950_THETA_RAD)
    @ rotation_axis_3(-B1950_ZETA_RAD)
)
J2000_TO_B1950 = B1950_TO_J2000.T


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one Earth-centred EME2000/J2000 trajectory CSV to "
            "SPENVIS GEI (B1950) upload format."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        help="Input trajectory CSV. Overrides INPUT_CSV in the script.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output SPENVIS file path.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="SPENVIS Title header. Defaults to the input filename stem.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    input_path = args.input_csv if args.input_csv is not None else INPUT_CSV
    input_path = input_path.expanduser()

    if str(input_path) == r"C:\path\to\trajectory.csv":
        raise ValueError(
            "Set INPUT_CSV near the top of the script or pass the CSV path "
            "as a command-line argument."
        )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    output_path = args.output if args.output is not None else OUTPUT_FILE
    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_spenvis_gei_b1950.txt"
        )
    output_path = output_path.expanduser()

    title = args.title or SPENVIS_TITLE or input_path.stem
    title = " ".join(str(title).splitlines()).strip()
    if not title:
        raise ValueError("SPENVIS title cannot be blank.")

    return input_path, output_path, title


def load_and_validate(input_path: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dataframe = pd.read_csv(input_path)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe]
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing)
        )

    if dataframe.empty:
        raise ValueError("Input CSV contains no trajectory rows.")

    # The source strings are UTC even though they do not include a trailing Z.
    utc = pd.to_datetime(dataframe["utc"], utc=True, errors="raise")
    utc_index = pd.DatetimeIndex(utc)

    if utc_index.has_duplicates:
        duplicate_count = int(utc_index.duplicated().sum())
        raise ValueError(f"UTC column contains {duplicate_count} duplicate epoch(s).")

    if not utc_index.is_monotonic_increasing:
        raise ValueError(
            "UTC epochs are not strictly increasing. The script does not sort "
            "rows because the original trajectory ordering is being preserved."
        )

    for column in REQUIRED_COLUMNS[1:]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")

    positions = dataframe.loc[:, REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(positions).all():
        bad_rows = np.flatnonzero(~np.isfinite(positions).all(axis=1))
        preview = ", ".join(str(int(index)) for index in bad_rows[:10])
        raise ValueError(
            "Position columns contain NaN or infinite values at zero-based "
            f"row index/indices: {preview}"
        )

    # Confirm that the hard-coded frame conversion remains a proper rotation.
    identity_error = np.max(
        np.abs(J2000_TO_B1950 @ J2000_TO_B1950.T - np.eye(3))
    )
    determinant = float(np.linalg.det(J2000_TO_B1950))
    if identity_error > 1.0e-12 or abs(determinant - 1.0) > 1.0e-12:
        raise RuntimeError("J2000-to-B1950 matrix failed its rotation check.")

    return dataframe, utc_index


def convert_positions(dataframe: pd.DataFrame) -> np.ndarray:
    positions_j2000_km = dataframe.loc[:, REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)

    # Each trajectory position is stored as a row vector. For column-vector
    # notation r_B1950 = M r_J2000, the equivalent row-vector operation is
    # r_B1950(row) = r_J2000(row) M^T.
    return positions_j2000_km @ J2000_TO_B1950.T


def format_second(timestamp: pd.Timestamp) -> str:
    second = (
        timestamp.second
        + timestamp.microsecond / 1.0e6
        + timestamp.nanosecond / 1.0e9
    )
    width = 2 + 1 + SECOND_DECIMALS
    return f"{second:0{width}.{SECOND_DECIMALS}f}"


def write_spenvis_file(
    output_path: Path,
    title: str,
    utc: pd.DatetimeIndex,
    positions_b1950_km: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    position_format = f"{{:.{POSITION_DECIMALS}f}}"

    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("* Converted from geocentric EME2000/J2000 to SPENVIS GEI B1950.\n")
        stream.write("* Input sampling and epochs are preserved. Position units are km.\n")
        stream.write(f"Title: {title}\n")
        stream.write("Planet: Earth\n")
        stream.write("Coordinates: GEI\n")
        stream.write("Columns: DYEA, DMON, DDAY, DHOU, DMIN, DSEC, X, Y, Z\n")
        stream.write("Format: CSV\n")
        stream.write("$$BEGIN\n")

        for timestamp, (x_km, y_km, z_km) in zip(utc, positions_b1950_km):
            stream.write(
                f"{timestamp.year:04d}, "
                f"{timestamp.month:02d}, "
                f"{timestamp.day:02d}, "
                f"{timestamp.hour:02d}, "
                f"{timestamp.minute:02d}, "
                f"{format_second(timestamp)}, "
                f"{position_format.format(x_km)}, "
                f"{position_format.format(y_km)}, "
                f"{position_format.format(z_km)}\n"
            )

        stream.write("$$END\n")


def main() -> int:
    args = parse_arguments()

    try:
        input_path, output_path, title = resolve_paths(args)
        dataframe, utc = load_and_validate(input_path)
        positions_b1950_km = convert_positions(dataframe)
        write_spenvis_file(output_path, title, utc, positions_b1950_km)
    except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    elapsed_days = (utc[-1] - utc[0]).total_seconds() / 86400.0
    input_positions = dataframe.loc[:, REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)
    input_radii = np.linalg.norm(input_positions, axis=1)

    print("SPENVIS trajectory conversion complete")
    print("--------------------------------------")
    print(f"Input:               {input_path.resolve()}")
    print(f"Output:              {output_path.resolve()}")
    print(f"Rows written:        {len(dataframe):,}")
    print(f"First UTC epoch:     {utc[0].isoformat()}")
    print(f"Last UTC epoch:      {utc[-1].isoformat()}")
    print(f"Elapsed length:      {elapsed_days:.9f} days")
    print(f"Minimum Earth range: {input_radii.min():.6f} km")
    print(f"Maximum Earth range: {input_radii.max():.6f} km")
    print()
    print(
        "Use the elapsed length above for the SPENVIS mission length, "
        "converting it to the units requested by the interface."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
