"""
Find representative cases nearest requested ΔV percentiles.

Run directly in PyCharm using the green Run button.
Edit the CONFIGURATION section below before running.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV = Path(
    r"results/transfer_results/transfer_90d/four_burn_dispersion_output_mpi/four_burn_optimized_sweep_summary.csv"
)

# Requested percentiles of the populated dv_total_mps distribution.
PERCENTILES = (15, 30, 45)

# Output file. Set to None to disable CSV output.
OUTPUT_CSV = Path(
    r"dv_percentile_cases.csv"
)

# Rows are considered populated when dv_total_mps can be converted to a finite
# numeric value. No optimizer_success or target_success filtering is applied.
DV_COLUMN = "dv_total_mps"
CASE_ID_COLUMN = "sample_id"

# Tie-breaking:
#   1. smallest absolute distance from the percentile value
#   2. lower dv_total_mps
#   3. lexicographically smaller case_id
# =============================================================================


def validate_percentiles(percentiles):
    """Validate and normalize percentile inputs."""
    values = []

    for percentile in percentiles:
        value = float(percentile)

        if not 0.0 <= value <= 100.0:
            raise ValueError(
                f"Percentile {percentile} is outside the valid range [0, 100]."
            )

        values.append(value)

    if not values:
        raise ValueError("PERCENTILES must contain at least one value.")

    return values


def load_valid_cases(input_csv):
    """Load rows having populated, finite total-ΔV values."""
    input_csv = Path(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    missing = [
        column
        for column in (CASE_ID_COLUMN, DV_COLUMN)
        if column not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df[DV_COLUMN] = pd.to_numeric(df[DV_COLUMN], errors="coerce")
    df = df[np.isfinite(df[DV_COLUMN])].copy()

    if df.empty:
        raise RuntimeError(
            f"No populated finite values were found in '{DV_COLUMN}'."
        )

    df[CASE_ID_COLUMN] = df[CASE_ID_COLUMN].astype(str)

    return df


def find_nearest_percentile_cases(df, percentiles):
    """
    For each requested percentile, find the actual row whose dv_total_mps is
    closest to the percentile value of the populated distribution.
    """
    dv_values = df[DV_COLUMN].to_numpy(dtype=float)
    results = []

    for percentile in percentiles:
        percentile_value = float(np.percentile(dv_values, percentile))

        candidates = df[[CASE_ID_COLUMN, DV_COLUMN]].copy()
        candidates["absolute_difference_mps"] = np.abs(
            candidates[DV_COLUMN] - percentile_value
        )

        selected = candidates.sort_values(
            by=[
                "absolute_difference_mps",
                DV_COLUMN,
                CASE_ID_COLUMN,
            ],
            ascending=[True, True, True],
            kind="mergesort",
        ).iloc[0]

        results.append(
            {
                "requested_percentile": percentile,
                "percentile_value_mps": percentile_value,
                "case_id": selected[CASE_ID_COLUMN],
                "case_dv_total_mps": float(selected[DV_COLUMN]),
                "absolute_difference_mps": float(
                    selected["absolute_difference_mps"]
                ),
            }
        )

    return pd.DataFrame(results)


def print_results(results_df, n_valid_cases):
    """Print a compact summary to the PyCharm console."""
    print()
    print("ΔV percentile case selection")
    print("----------------------------")
    print(f"Valid populated cases: {n_valid_cases}")
    print()

    for row in results_df.itertuples(index=False):
        print(
            f"{row.requested_percentile:g}th percentile: "
            f"target={row.percentile_value_mps:.6f} m/s, "
            f"case_id={row.case_id}, "
            f"case ΔV={row.case_dv_total_mps:.6f} m/s, "
            f"|difference|={row.absolute_difference_mps:.6f} m/s"
        )


def main():
    percentiles = validate_percentiles(PERCENTILES)
    valid_df = load_valid_cases(INPUT_CSV)

    results_df = find_nearest_percentile_cases(
        df=valid_df,
        percentiles=percentiles,
    )

    print_results(
        results_df=results_df,
        n_valid_cases=len(valid_df),
    )

    if OUTPUT_CSV is not None:
        output_path = Path(OUTPUT_CSV)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

        print()
        print(f"Saved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
