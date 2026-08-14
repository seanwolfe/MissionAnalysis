import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = Path(r"to_sync/mission_summary/MASTER_IOD.csv")

# Sliding-window settings, in days
WINDOW_LENGTH_DAYS = 365.25 * 5
SLIDE_DAYS = 180.0

# Mission success criterion:
# Minimum number of converged cases required within the window
MISSION_SUCCESS_THRESHOLD = 3

# A run is considered converged when this appears in
# OD_TERMINATION_REASON
CONVERGENCE_REASON = "convergence_criteria_met"

# Save individual sliding-window results
SAVE_WINDOW_CSV = True
OUTPUT_CSV = Path("sliding_window_convergence.csv")

# Plot
MAKE_PLOT = True
OUTPUT_PLOT = Path("sliding_window_convergence.svg")


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

epoch_min = df["EPOCH_AST(jdtdb)"].min()
epoch_max = df["EPOCH_AST(jdtdb)"].max()

print()
print("Dataset epoch range")
print("-------------------")
print(f"Start JDTDB : {epoch_min:.6f}")
print(f"End JDTDB   : {epoch_max:.6f}")
print(f"Duration    : {epoch_max - epoch_min:.2f} days")

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
# SLIDING WINDOW
# ============================================================

# Only include complete mission-duration windows.
#
# Therefore, every point represents:
#
#   mission start = start
#   mission end   = start + WINDOW_LENGTH_DAYS
#
# This avoids artificially low counts for mission starts near
# the end of the simulated interval.

last_window_start = epoch_max - WINDOW_LENGTH_DAYS

if last_window_start < epoch_min:
    raise ValueError(
        "WINDOW_LENGTH_DAYS is longer than the total dataset."
    )

window_starts = np.arange(
    epoch_min,
    last_window_start + 1e-12,
    SLIDE_DAYS
)

results = []

for start in window_starts:

    end = start + WINDOW_LENGTH_DAYS

    # Include:
    # start <= EPOCH_AST < end
    mask = (
        (df["EPOCH_AST(jdtdb)"] >= start) &
        (df["EPOCH_AST(jdtdb)"] < end)
    )

    window = df.loc[mask]

    n_total = len(window)
    n_converged = int(window["CONVERGED"].sum())

    if n_total > 0:
        convergence_fraction = n_converged / n_total
    else:
        convergence_fraction = np.nan

    mission_success = (
        n_converged >= MISSION_SUCCESS_THRESHOLD
    )

    results.append({
        "mission_start_jdtdb": start,
        "mission_end_jdtdb": end,
        "window_length_days": WINDOW_LENGTH_DAYS,
        "n_total_scenarios": n_total,
        "n_converged": n_converged,
        "convergence_fraction": convergence_fraction,
        "mission_success": mission_success,
    })


windows = pd.DataFrame(results)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

counts = windows["n_converged"]

print()
print("Sliding-window results")
print("----------------------")
print(f"Mission duration : {WINDOW_LENGTH_DAYS:.2f} days")
print(f"Slide            : {SLIDE_DAYS:.2f} days")
print(f"No. windows      : {len(windows):,}")

print()
print("Converged scenarios per mission window")
print("--------------------------------------")
print(f"Mean   : {counts.mean():.3f}")
print(f"Median : {counts.median():.3f}")
print(f"Std    : {counts.std():.3f}")
print(f"Min    : {counts.min():.0f}")
print(f"Max    : {counts.max():.0f}")

print()
print("Percentiles")
print("-----------")

for p in [5, 10, 25, 50, 75, 90, 95]:
    value = np.percentile(counts, p)
    print(f"{p:>2}th percentile : {value:.2f}")


# ============================================================
# MISSION SUCCESS PROBABILITY
# ============================================================

success_probability = windows["mission_success"].mean()

print()
print("Mission success")
print("---------------")
print(
    f"Criterion : >= {MISSION_SUCCESS_THRESHOLD} "
    f"converged scenarios"
)
print(
    f"P(success) = "
    f"{100 * success_probability:.2f}%"
)


# Also provide probabilities for several thresholds
print()
print("Probability of at least N converged scenarios")
print("---------------------------------------------")

for n in [1, 2, 3, 4, 5]:
    probability = np.mean(counts >= n)

    print(
        f"P(N_converged >= {n}) = "
        f"{100 * probability:.2f}%"
    )


# ============================================================
# SAVE WINDOW DATA
# ============================================================

if SAVE_WINDOW_CSV:

    windows.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print(f"Saved: {OUTPUT_CSV}")


# ============================================================
# PLOT
# ============================================================

if MAKE_PLOT:

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    # Number of converged scenarios for a mission beginning
    # at each candidate mission-start epoch
    ax.plot(
        windows["mission_start_jdtdb"],
        windows["n_converged"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Converged scenarios",
    )

    # Mean
    ax.axhline(
        counts.mean(),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {counts.mean():.2f}",
    )

    # Mission-success requirement
    ax.axhline(
        MISSION_SUCCESS_THRESHOLD,
        color="red",
        linestyle="-",
        linewidth=2,
        label=(
            f"Mission success = "
            f"{MISSION_SUCCESS_THRESHOLD}"
        ),
    )

    ax.set_xlabel(
        "Nominal mission operations start epoch (JDTDB)"
    )

    ax.set_ylabel(
        f"Converged scenarios within "
        f"{WINDOW_LENGTH_DAYS:g} days"
    )

    ax.grid(alpha=0.25)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PLOT,
        bbox_inches="tight"
    )

    print(f"Saved: {OUTPUT_PLOT}")

    plt.show()