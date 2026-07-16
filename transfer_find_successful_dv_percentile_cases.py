"""
Find representative successful cases nearest requested ΔV percentiles.

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
    r"results/transfer_results/transfer_90d/four_burn_dispersion_output_mpi/"
    r"four_burn_optimized_sweep_summary.csv"
)

# Requested percentiles of the successful, populated dv_total_mps distribution.
PERCENTILES = (15, 30, 45)

# Output file. Set to None to disable CSV output.
OUTPUT_CSV = Path(
    r"dv_percentile_cases.csv"
)

DV_COLUMN = "dv_total_mps"
CASE_ID_COLUMN = "sample_id"
OPTIMIZER_SUCCESS_COLUMN = "optimizer_success"
TARGET_SUCCESS_COLUMN = "target_success"

# Only rows satisfying BOTH conditions are included:
#   optimizer_success == True
#   target_success == True
#
# Among those rows, dv_total_mps must also be finite and populated.
#
# Tie-breaking:
#   1. smallest absolute distance from the percentile value
#   2. lower dv_total_mps
#   3. lexicographically smaller case identifier
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


def normalize_boolean_column(series):
    """
    Convert common CSV boolean representations to pandas nullable booleans.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    normalized = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "nan": pd.NA,
        "none": pd.NA,
        "": pd.NA,
    }

    return normalized.map(mapping).astype("boolean")


def load_valid_cases(input_csv):
    """
    Load cases satisfying both success flags and having finite total ΔV.
    """
    input_csv = Path(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_columns = (
        CASE_ID_COLUMN,
        DV_COLUMN,
        OPTIMIZER_SUCCESS_COLUMN,
        TARGET_SUCCESS_COLUMN,
    )

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()

    df[OPTIMIZER_SUCCESS_COLUMN] = normalize_boolean_column(
        df[OPTIMIZER_SUCCESS_COLUMN]
    )
    df[TARGET_SUCCESS_COLUMN] = normalize_boolean_column(
        df[TARGET_SUCCESS_COLUMN]
    )

    n_total = len(df)

    success_mask = (
        df[OPTIMIZER_SUCCESS_COLUMN].fillna(False)
        & df[TARGET_SUCCESS_COLUMN].fillna(False)
    )
    df = df[success_mask].copy()

    n_success = len(df)

    df[DV_COLUMN] = pd.to_numeric(df[DV_COLUMN], errors="coerce")
    df = df[np.isfinite(df[DV_COLUMN])].copy()

    n_valid = len(df)

    if df.empty:
        raise RuntimeError(
            "No rows remained after filtering for optimizer_success == True, "
            "target_success == True, and finite dv_total_mps."
        )

    df[CASE_ID_COLUMN] = df[CASE_ID_COLUMN].astype(str)

    counts = {
        "n_total_rows": n_total,
        "n_success_rows": n_success,
        "n_valid_rows": n_valid,
    }

    return df, counts


def find_nearest_percentile_cases(df, percentiles):
    """
    For each requested percentile, find the actual successful row whose
    dv_total_mps is closest to that percentile value.
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
                CASE_ID_COLUMN: selected[CASE_ID_COLUMN],
                "case_dv_total_mps": float(selected[DV_COLUMN]),
                "absolute_difference_mps": float(
                    selected["absolute_difference_mps"]
                ),
            }
        )

    return pd.DataFrame(results)


def print_results(results_df, counts):
    """Print a compact summary to the PyCharm console."""
    print()
    print("Successful ΔV percentile case selection")
    print("----------------------------------------")
    print(f"Total input rows:                    {counts['n_total_rows']}")
    print(f"Optimizer + target success rows:     {counts['n_success_rows']}")
    print(f"Successful rows with populated ΔV:   {counts['n_valid_rows']}")
    print()

    for _, row in results_df.iterrows():
        print(
            f"{row['requested_percentile']:g}th percentile: "
            f"target={row['percentile_value_mps']:.6f} m/s, "
            f"{CASE_ID_COLUMN}={row[CASE_ID_COLUMN]}, "
            f"case ΔV={row['case_dv_total_mps']:.6f} m/s, "
            f"|difference|={row['absolute_difference_mps']:.6f} m/s"
        )


def main():
    percentiles = validate_percentiles(PERCENTILES)
    valid_df, counts = load_valid_cases(INPUT_CSV)

    results_df = find_nearest_percentile_cases(
        df=valid_df,
        percentiles=percentiles,
    )

    print_results(
        results_df=results_df,
        counts=counts,
    )

    if OUTPUT_CSV is not None:
        output_path = Path(OUTPUT_CSV)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

        print()
        print(f"Saved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
