import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = Path(r"to_sync\to_sync\phase_results\copy_5_2\MASTER_IOD.csv")

# Nominal-operations-duration sweep, in days
MIN_OPERATIONS_DURATION_DAYS = 365.0
MAX_OPERATIONS_DURATION_DAYS = 10 * 365.25
OPERATIONS_DURATION_STEP_DAYS = 180.0

# Sliding-window step, in days
SLIDE_DAYS = 180.0

# Mission success criterion:
# Minimum number of identified/converged TBOs required within a window
MISSION_SUCCESS_THRESHOLD = 3

# A run is considered converged when this appears in
# OD_TERMINATION_REASON
CONVERGENCE_REASON = "convergence_criteria_met"

# Save results
SAVE_SUMMARY_CSV = True
SUMMARY_OUTPUT_CSV = Path("mission_success_duration_sweep_summary.csv")

# Optional: save every individual sliding window from every duration
SAVE_WINDOW_CSV = False
WINDOW_OUTPUT_CSV = Path("mission_success_duration_sweep_windows.csv")

# Plots
MAKE_PLOTS = True
OUTPUT_MEAN_TBO_PLOT = Path("window_length_vs_mean_tbos_identified.svg")
OUTPUT_SUCCESS_PLOT = Path("window_length_vs_mission_success_probability.svg")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

required_columns = [
    "EPOCH_AST(jdtdb)",
    "OD_TERMINATION_REASON",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required column(s): {missing}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df["EPOCH_AST(jdtdb)"] = pd.to_numeric(
    df["EPOCH_AST(jdtdb)"],
    errors="coerce"
)

df = df.dropna(subset=["EPOCH_AST(jdtdb)"]).copy()

termination_reason = (
    df["OD_TERMINATION_REASON"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

df["CONVERGED"] = (
    termination_reason == CONVERGENCE_REASON.lower()
)


# ============================================================
# OPTIONAL DUPLICATE HANDLING
#
# If ID_AST / RUN_NUMBER combinations represent the same
# physical scenario and appear more than once, uncomment:
# ============================================================

# df = (
#     df.groupby(["ID_AST", "RUN_NUMBER"], as_index=False)
#     .agg({
#         "EPOCH_AST(jdtdb)": "first",
#         "CONVERGED": "max",
#     })
# )


# ============================================================
# SORT BY EPOCH
# ============================================================

df = df.sort_values(
    "EPOCH_AST(jdtdb)"
).reset_index(drop=True)

if df.empty:
    raise ValueError("No valid rows remain after cleaning the epoch column.")

epoch_min = df["EPOCH_AST(jdtdb)"].min()
epoch_max = df["EPOCH_AST(jdtdb)"].max()
dataset_duration = epoch_max - epoch_min

print()
print("Dataset epoch range")
print("-------------------")
print(f"Start JDTDB : {epoch_min:.6f}")
print(f"End JDTDB   : {epoch_max:.6f}")
print(f"Duration    : {dataset_duration:.2f} days")

print()
print("Overall results")
print("---------------")
print(f"Total scenarios     : {len(df):,}")
print(f"Converged scenarios : {df['CONVERGED'].sum():,}")
print(
    f"Overall convergence : "
    f"{100 * df['CONVERGED'].mean():.2f}%"
)


# ============================================================
# BUILD OPERATIONS-DURATION SWEEP
# ============================================================

if MIN_OPERATIONS_DURATION_DAYS <= 0:
    raise ValueError("MIN_OPERATIONS_DURATION_DAYS must be > 0.")

if MAX_OPERATIONS_DURATION_DAYS < MIN_OPERATIONS_DURATION_DAYS:
    raise ValueError(
        "MAX_OPERATIONS_DURATION_DAYS must be >= "
        "MIN_OPERATIONS_DURATION_DAYS."
    )

if OPERATIONS_DURATION_STEP_DAYS <= 0:
    raise ValueError("OPERATIONS_DURATION_STEP_DAYS must be > 0.")

if SLIDE_DAYS <= 0:
    raise ValueError("SLIDE_DAYS must be > 0.")

operations_durations = np.arange(
    MIN_OPERATIONS_DURATION_DAYS,
    MAX_OPERATIONS_DURATION_DAYS + 1e-12,
    OPERATIONS_DURATION_STEP_DAYS,
    dtype=float,
)

# Include the user-specified maximum exactly, even when the regular
# step sequence does not land on it (e.g. 365 -> 5*365.25 in 30-day steps).
if (
    len(operations_durations) == 0
    or operations_durations[-1] < MAX_OPERATIONS_DURATION_DAYS - 1e-9
):
    operations_durations = np.append(
        operations_durations,
        MAX_OPERATIONS_DURATION_DAYS,
    )

# Only complete windows can be evaluated.
valid_durations = operations_durations[
    operations_durations <= dataset_duration + 1e-9
]

if len(valid_durations) == 0:
    raise ValueError(
        "The minimum requested operations duration is longer than "
        "the total dataset duration."
    )

if len(valid_durations) < len(operations_durations):
    print()
    print(
        "Warning: requested operations durations longer than the "
        "dataset were skipped."
    )

print()
print("Duration sweep")
print("--------------")
print(f"Minimum duration : {valid_durations.min():.2f} days")
print(f"Maximum duration : {valid_durations.max():.2f} days")
print(f"Nominal step     : {OPERATIONS_DURATION_STEP_DAYS:.2f} days")
print(f"Window slide     : {SLIDE_DAYS:.2f} days")
print(f"No. durations    : {len(valid_durations):,}")


# ============================================================
# SLIDING-WINDOW ANALYSIS FOR EACH OPERATIONS DURATION
# ============================================================

# For each candidate nominal operations duration:
#   1. Form every complete sliding window of that length.
#   2. Count the identified/converged TBOs in every window.
#   3. Compute the mean identified-TBO count.
#   4. Compute the proportion of windows satisfying the mission
#      success requirement.

summary_results = []
all_window_results = []

for duration in valid_durations:

    last_window_start = epoch_max - duration

    window_starts = np.arange(
        epoch_min,
        last_window_start + 1e-12,
        SLIDE_DAYS,
        dtype=float,
    )

    counts = []
    successes = []

    for start in window_starts:

        end = start + duration

        # Half-open interval:
        # start <= EPOCH_AST < end
        mask = (
            (df["EPOCH_AST(jdtdb)"] >= start)
            & (df["EPOCH_AST(jdtdb)"] < end)
        )

        window = df.loc[mask]

        n_total = len(window)
        n_converged = int(window["CONVERGED"].sum())
        mission_success = (
            n_converged >= MISSION_SUCCESS_THRESHOLD
        )

        counts.append(n_converged)
        successes.append(mission_success)

        if SAVE_WINDOW_CSV:
            all_window_results.append({
                "nominal_operations_duration_days": duration,
                "mission_start_jdtdb": start,
                "mission_end_jdtdb": end,
                "n_total_scenarios": n_total,
                "n_tbos_identified": n_converged,
                "mission_success": mission_success,
            })

    counts = np.asarray(counts, dtype=float)
    successes = np.asarray(successes, dtype=bool)

    summary_results.append({
        "nominal_operations_duration_days": duration,
        "n_complete_windows": len(counts),
        "mean_n_tbos_identified": counts.mean(),
        "median_n_tbos_identified": np.median(counts),
        "std_n_tbos_identified": counts.std(ddof=1) if len(counts) > 1 else 0.0,
        "min_n_tbos_identified": counts.min(),
        "max_n_tbos_identified": counts.max(),
        "mission_success_probability": successes.mean(),
    })

summary = pd.DataFrame(summary_results)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("Sweep results")
print("-------------")
print(
    f"Mission success criterion: >= {MISSION_SUCCESS_THRESHOLD} "
    f"identified TBOs"
)
print()

print(
    summary[
        [
            "nominal_operations_duration_days",
            "n_complete_windows",
            "mean_n_tbos_identified",
            "mission_success_probability",
        ]
    ].to_string(
        index=False,
        formatters={
            "nominal_operations_duration_days": lambda x: f"{x:.2f}",
            "mean_n_tbos_identified": lambda x: f"{x:.3f}",
            "mission_success_probability": lambda x: f"{x:.4f}",
        },
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

if SAVE_SUMMARY_CSV:
    summary.to_csv(
        SUMMARY_OUTPUT_CSV,
        index=False,
    )
    print()
    print(f"Saved: {SUMMARY_OUTPUT_CSV}")

if SAVE_WINDOW_CSV:
    all_windows = pd.DataFrame(all_window_results)
    all_windows.to_csv(
        WINDOW_OUTPUT_CSV,
        index=False,
    )
    print(f"Saved: {WINDOW_OUTPUT_CSV}")


# ============================================================
# PLOTS
# ============================================================

if MAKE_PLOTS:

    # --------------------------------------------------------
    # Plot 1: operations duration vs mean identified TBOs
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        summary["nominal_operations_duration_days"] / 365.25,
        summary["mean_n_tbos_identified"],
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    ax.set_xlabel("Nominal Operations Duration (years)")
    ax.set_ylabel("Mean Number of TBOs Identified")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_MEAN_TBO_PLOT,
        bbox_inches="tight",
    )

    print(f"Saved: {OUTPUT_MEAN_TBO_PLOT}")

    # --------------------------------------------------------
    # Plot 2: operations duration vs mission success probability
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        summary["nominal_operations_duration_days"] / 365.25,
        summary["mission_success_probability"],
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    ax.set_xlabel("Nominal Operations Duration (years)")
    ax.set_ylabel("Mission Success Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_SUCCESS_PLOT,
        bbox_inches="tight",
    )

    print(f"Saved: {OUTPUT_SUCCESS_PLOT}")

    plt.show()
