from __future__ import annotations

"""Summarize mission detection, IOD, OD, loss, and EMS outcomes.

This module is intentionally independent of the numerical simulation modules.
It reads the existing visible-detection files, ``MASTER_IOD.csv``, OD outer-loop
CSV files, and OD progress records. It can be imported by the overall
simulation or run directly from the command line.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping
import json
import os
import re

import numpy as np
import pandas as pd


_TERMINAL_EVENT_TO_OUTCOME = {
    "termination_convergence": "converged",
    "termination_object_lost_non_ems_no_detection": "lost_non_ems",
    "termination_no_detection": "lost_non_ems",
    "termination_object_lost_after_ems_reacquisition": (
        "lost_after_ems_reacquisition"
    ),
    "termination_time_limit": "time_limit",
    "termination_step_limit": "step_limit",
    "termination_error": "runtime_error",
    "termination_attcoord_grid_too_short": "model_horizon_limit",
    "termination_minimoon_orbit_grid_exhausted": "model_horizon_limit",
    "termination_minimoon_orbit_grid_invalid": "model_horizon_limit",
    "termination_minimoon_orbit_grid_nearly_exhausted": (
        "model_horizon_limit"
    ),
    "termination_piecewise_target_grid_empty": "model_horizon_limit",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(int(value))
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "t", "on"
    }


def _finite_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    return number if np.isfinite(number) else float(default)


def _valid_six_state(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return False
    parts = [
        part for part in re.split(r"[,;\s\[\]\(\)]+", text)
        if part
    ]
    try:
        state = np.asarray([float(part) for part in parts], dtype=float)
    except Exception:
        return False
    return state.size == 6 and bool(np.all(np.isfinite(state)))


def _last_finite(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or len(frame) == 0:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.iloc[-1]) if len(values) else np.nan


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_visible_tables(visible_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not visible_dir.is_dir():
        return pd.DataFrame()

    paths = sorted(
        list(visible_dir.glob("*.csv"))
        + list(visible_dir.glob("*.parquet"))
    )
    for path in paths:
        try:
            frame = _read_table(path)
        except Exception as exc:
            print(
                f"[Mission Summary] Skipping unreadable visible file "
                f"{path}: {exc}",
                flush=True,
            )
            continue
        required = {"run_number", "object_id", "spacecraft_number"}
        if required.issubset(frame.columns):
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in ("run_number", "spacecraft_number"):
        combined[column] = pd.to_numeric(
            combined[column], errors="coerce"
        )
    combined = combined.dropna(
        subset=["run_number", "object_id", "spacecraft_number"]
    )
    combined["run_number"] = combined["run_number"].astype(int)
    combined["spacecraft_number"] = (
        combined["spacecraft_number"].astype(int)
    )
    combined["object_id"] = combined["object_id"].astype(str)

    # CSV and parquet versions of the same source may both exist.
    combined = combined.drop_duplicates(
        subset=["run_number", "object_id", "spacecraft_number"],
        keep="last",
    )
    return combined.reset_index(drop=True)


def _infer_master_run_numbers(
    master: pd.DataFrame,
    *,
    number_of_runs: int,
) -> pd.Series:
    if "RUN_NUMBER" in master.columns:
        values = pd.to_numeric(master["RUN_NUMBER"], errors="coerce")
    else:
        values = pd.Series(np.nan, index=master.index, dtype=float)

    filename_columns = [
        column for column in (
            "IOD_DATA_SAVED_AS",
            "OD_RUN_UID",
            "OD_RESULT_SAVED_AS",
        )
        if column in master.columns
    ]
    for row_index in master.index[~np.isfinite(values)]:
        joined = " ".join(
            str(master.at[row_index, column])
            for column in filename_columns
        )
        match = re.search(r"(?:^|[_-])run-(\d+)(?:[_-]|$)", joined)
        if match:
            values.at[row_index] = int(match.group(1))

    if int(number_of_runs) == 1:
        values = values.fillna(1)

    return values


def _choose_outer_path(
    master_row: pd.Series,
    *,
    master_row_index: int,
    top_dir: Path,
    outer_by_master_row: Mapping[int, Path],
) -> Path | None:
    raw = master_row.get("OD_OUTER_CSV_PATH", "")
    if raw is not None and str(raw).strip() and str(raw).lower() != "nan":
        candidate = Path(str(raw)).expanduser()
        candidates = [candidate]
        if not candidate.is_absolute():
            candidates.extend([top_dir / candidate, Path.cwd() / candidate])
        for value in candidates:
            if value.is_file():
                return value.resolve()

    return outer_by_master_row.get(int(master_row_index))


def _index_outer_files(outer_dir: Path) -> dict[int, Path]:
    """Map MASTER row index to the strongest available outer-loop file."""

    candidates: dict[int, list[tuple[int, float, Path]]] = {}
    if not outer_dir.is_dir():
        return {}

    for path in sorted(outer_dir.glob("*__outer.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(
                f"[Mission Summary] Skipping unreadable OD outer file "
                f"{path}: {exc}",
                flush=True,
            )
            continue
        if len(frame) == 0 or "master_row_idx" not in frame.columns:
            continue
        try:
            row_index = int(
                pd.to_numeric(
                    frame["master_row_idx"], errors="coerce"
                ).dropna().iloc[0]
            )
        except Exception:
            continue

        event = (
            frame["event_type"].fillna("").astype(str)
            if "event_type" in frame.columns
            else pd.Series("", index=frame.index)
        )
        terminal_score = int(event.str.startswith("termination_").any())
        candidates.setdefault(row_index, []).append(
            (terminal_score, path.stat().st_mtime, path.resolve())
        )

    return {
        row_index: sorted(values, key=lambda item: (item[0], item[1]))[-1][2]
        for row_index, values in candidates.items()
    }


def _terminal_outcome(
    outer: pd.DataFrame,
    *,
    progress: Mapping[str, Any] | None,
) -> tuple[str, str, str]:
    if len(outer) > 0 and "event_type" in outer.columns:
        events = outer["event_type"].fillna("").astype(str)
        terminal_rows = outer[events.str.startswith("termination_")]
        if len(terminal_rows) > 0:
            final = terminal_rows.iloc[-1]
            event = str(final.get("event_type", "")).strip()
            reason = str(final.get("termination_reason", "")).strip()
            outcome = _TERMINAL_EVENT_TO_OUTCOME.get(
                event, "other_termination"
            )
            if event == "termination_no_detection" and (
                "after_ems" in reason.lower()
                or "reacquisition" in reason.lower()
            ):
                outcome = "lost_after_ems_reacquisition"
            return outcome, event, reason

    if progress:
        reason = str(progress.get("termination_reason", "")).strip()
        if reason == "walltime_checkpoint":
            return "paused_checkpoint", "", reason
        if bool(progress.get("completed", False)):
            return "completed_unknown", "", reason

    if len(outer) > 0:
        return "incomplete", "", ""
    return "od_not_started", "", ""


def _blackout_statistics(outer: pd.DataFrame) -> dict[str, Any]:
    if len(outer) == 0:
        return {
            "ever_ems_blackout": False,
            "n_ems_blackout_episodes": 0,
            "total_ems_blackout_duration_s": 0.0,
            "max_ems_blackout_duration_s": 0.0,
            "n_reacquisition_attempts": 0,
            "n_successful_reacquisitions": 0,
            "n_prediction_only_cycles": 0,
        }

    event = (
        outer["event_type"].fillna("").astype(str)
        if "event_type" in outer.columns
        else pd.Series("", index=outer.index)
    )
    blackout = event.eq("ems_blackout_prediction_only").to_numpy(dtype=bool)

    # Retain compatibility with older logs that did not use the dedicated
    # event name but did record the all-EMS-occluded flag.
    if "all_ems_occluded" in outer.columns:
        all_ems = outer["all_ems_occluded"].map(_as_bool).to_numpy(bool)
        blackout = blackout | all_ems

    episode_starts = np.flatnonzero(
        blackout & np.concatenate(([True], ~blackout[:-1]))
    )
    episode_ends = np.flatnonzero(
        blackout & np.concatenate((~blackout[1:], [True]))
    )

    row_durations = np.zeros(len(outer), dtype=float)
    if {
        "epoch_start_jdtdb",
        "epoch_end_jdtdb",
    }.issubset(outer.columns):
        start = pd.to_numeric(
            outer["epoch_start_jdtdb"], errors="coerce"
        ).to_numpy(float)
        end = pd.to_numeric(
            outer["epoch_end_jdtdb"], errors="coerce"
        ).to_numpy(float)
        row_durations = np.where(
            np.isfinite(start) & np.isfinite(end) & (end >= start),
            (end - start) * 86400.0,
            0.0,
        )

    episode_durations = [
        float(np.sum(row_durations[start:end + 1]))
        for start, end in zip(episode_starts, episode_ends)
    ]

    detection = np.zeros(len(outer), dtype=bool)
    if "had_detection" in outer.columns:
        detection = outer["had_detection"].map(_as_bool).to_numpy(bool)
    detection &= event.eq("regular_update").to_numpy(bool)

    terminal = event.str.startswith("termination_").to_numpy(bool)
    successful_reacquisitions = 0
    for episode_number, end_index in enumerate(episode_ends):
        next_start = (
            int(episode_starts[episode_number + 1])
            if episode_number + 1 < len(episode_starts)
            else len(outer)
        )
        search_end = next_start
        later_terminal = np.flatnonzero(
            terminal[end_index + 1:next_start]
        )
        if later_terminal.size:
            search_end = end_index + 1 + int(later_terminal[0]) + 1
        if np.any(detection[end_index + 1:search_end]):
            successful_reacquisitions += 1

    if "reacquisition_attempt_count" in outer.columns:
        attempts_series = pd.to_numeric(
            outer["reacquisition_attempt_count"], errors="coerce"
        )
        attempts = (
            int(attempts_series.max())
            if attempts_series.notna().any()
            else 0
        )
    else:
        attempts = int(event.eq("ems_reacquisition_attcoord").sum())

    return {
        "ever_ems_blackout": bool(np.any(blackout)),
        "n_ems_blackout_episodes": int(len(episode_starts)),
        "total_ems_blackout_duration_s": float(
            np.sum(episode_durations)
        ),
        "max_ems_blackout_duration_s": float(
            max(episode_durations, default=0.0)
        ),
        "n_reacquisition_attempts": int(attempts),
        "n_successful_reacquisitions": int(successful_reacquisitions),
        "n_prediction_only_cycles": int(np.sum(blackout)),
    }


def _progress_for_case(
    *,
    detail_root: Path,
    run_uid: str,
) -> Mapping[str, Any] | None:
    if not run_uid:
        return None
    path = detail_root / run_uid / "progress.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, Mapping) else None
    except Exception:
        return None


def _case_summary_row(
    master_row: pd.Series,
    *,
    master_row_index: int,
    run_number: float,
    outer_path: Path | None,
    detail_root: Path,
) -> dict[str, Any]:
    iod_success = (
        _valid_six_state(master_row.get("IOD_FINAL_STATE", None))
        and bool(str(master_row.get("IOD_RESULT_SAVED_AS", "")).strip())
        and str(master_row.get("IOD_RESULT_SAVED_AS", "")).lower() != "nan"
    )

    outer = pd.DataFrame()
    if outer_path is not None:
        try:
            outer = pd.read_csv(outer_path)
        except Exception as exc:
            print(
                f"[Mission Summary] Could not read OD outer file "
                f"{outer_path}: {exc}",
                flush=True,
            )

    run_uid = ""
    if len(outer) and "run_uid" in outer.columns:
        nonempty = outer["run_uid"].dropna().astype(str)
        if len(nonempty):
            run_uid = str(nonempty.iloc[0])
    if not run_uid:
        run_uid = str(master_row.get("OD_RUN_UID", "") or "").strip()
    if not run_uid:
        saved_as = str(master_row.get("IOD_DATA_SAVED_AS", "") or "")
        first = saved_as.split(";")[0].strip()
        run_uid = Path(first).stem if first else ""

    progress = _progress_for_case(
        detail_root=detail_root,
        run_uid=run_uid,
    )

    od_started = len(outer) > 0
    if not iod_success:
        outcome, terminal_event, terminal_reason = (
            "iod_failed", "", ""
        )
    else:
        outcome, terminal_event, terminal_reason = _terminal_outcome(
            outer,
            progress=progress,
        )

    blackout = _blackout_statistics(outer)

    event = (
        outer["event_type"].fillna("").astype(str)
        if len(outer) and "event_type" in outer.columns
        else pd.Series(dtype=str)
    )
    n_od_updates = int(event.eq("regular_update").sum())
    n_detected_cycles = 0
    if len(outer) and "had_detection" in outer.columns:
        n_detected_cycles = int(
            (
                event.eq("regular_update")
                & outer["had_detection"].map(_as_bool)
            ).sum()
        )

    time_to_convergence_days = np.nan
    if outcome == "converged" and len(outer):
        start = _finite_float(
            pd.to_numeric(
                outer.get("epoch_start_jdtdb", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna().iloc[0]
            if "epoch_start_jdtdb" in outer.columns
            and pd.to_numeric(
                outer["epoch_start_jdtdb"], errors="coerce"
            ).notna().any()
            else np.nan
        )
        terminal_end = np.nan
        terminal_mask = event.eq("termination_convergence")
        if terminal_mask.any():
            terminal_end = _finite_float(
                outer.loc[
                    terminal_mask, "epoch_end_jdtdb"
                ].iloc[-1]
            )
        if np.isfinite(start) and np.isfinite(terminal_end):
            time_to_convergence_days = terminal_end - start

    object_id = str(master_row.get("ID_AST", ""))
    detecting_sc = _finite_float(
        master_row.get("DETECTING_SC_ID", np.nan)
    )
    index_used = _finite_float(master_row.get("INDEX_USED", np.nan))

    return {
        "master_row_idx": int(master_row_index),
        "run_number": (
            int(run_number) if np.isfinite(run_number) else np.nan
        ),
        "object_id": object_id,
        "tbo_H": _finite_float(master_row.get("TBO_H", np.nan)),
        "detecting_spacecraft_id": (
            int(detecting_sc) if np.isfinite(detecting_sc) else np.nan
        ),
        "index_used": (
            int(index_used) if np.isfinite(index_used) else np.nan
        ),
        "initially_detected": True,
        "iod_success": bool(iod_success),
        "od_started": bool(od_started),
        "outcome": outcome,
        "terminal_event": terminal_event,
        "terminal_reason": terminal_reason,
        "converged": outcome == "converged",
        "lost_non_ems": outcome == "lost_non_ems",
        "lost_after_ems_reacquisition": (
            outcome == "lost_after_ems_reacquisition"
        ),
        "time_limit": outcome == "time_limit",
        "step_limit": outcome == "step_limit",
        "runtime_error": outcome == "runtime_error",
        "incomplete": outcome in {
            "incomplete", "od_not_started", "paused_checkpoint"
        },
        **blackout,
        "n_od_updates": n_od_updates,
        "n_detected_cycles": n_detected_cycles,
        "time_to_convergence_days": time_to_convergence_days,
        "final_position_error_km": _last_finite(
            outer, "pos_err_norm"
        ),
        "final_velocity_error_km_s": _last_finite(
            outer, "vel_err_norm"
        ),
        "final_position_cov_trace_km2": _last_finite(
            outer, "P_pos_trace"
        ),
        "final_velocity_cov_trace_km2_s2": _last_finite(
            outer, "P_vel_trace"
        ),
        "final_nis_mean": _last_finite(outer, "NIS_mean"),
        "run_uid": run_uid,
        "outer_csv_path": (
            str(outer_path) if outer_path is not None else ""
        ),
    }


def _metric_row(
    metric: str,
    *,
    numerator: float | int | None = None,
    denominator: float | int | None = None,
    value: float | int | None = None,
    units: str = "count",
) -> dict[str, Any]:
    fraction = np.nan
    if (
        numerator is not None
        and denominator is not None
        and float(denominator) > 0.0
    ):
        fraction = float(numerator) / float(denominator)
    return {
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "fraction": fraction,
        "value": value,
        "units": units,
    }


def _campaign_statistics(
    visible: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    include_ems_exclusion: bool,
) -> pd.DataFrame:
    metrics: list[dict[str, Any]] = []

    visible_all_keys: set[tuple[int, str]] = set()
    visible_detected_keys: set[tuple[int, str]] = set()
    detected_spacecraft_cases = 0
    if len(visible):
        detection_column = (
            "n_detection_ems"
            if include_ems_exclusion
            and "n_detection_ems" in visible.columns
            else "n_detection"
        )
        active = pd.to_numeric(
            visible.get(detection_column, 0), errors="coerce"
        ).fillna(0) > 0
        visible_work = visible.assign(_active_detection=active)
        visible_all_keys = {
            (int(run), str(object_id))
            for run, object_id in zip(
                visible_work["run_number"],
                visible_work["object_id"],
            )
        }
        visible_detected_keys = {
            (int(row.run_number), str(row.object_id))
            for row in visible_work.loc[
                visible_work["_active_detection"],
                ["run_number", "object_id"],
            ].itertuples(index=False)
        }
        detected_spacecraft_cases = int(active.sum())

    # MASTER_IOD contains only initially detected cases. Including its keys
    # keeps statistics internally consistent if the summary is run against a
    # partial visible-file archive.
    case_keys: set[tuple[int, str]] = set()
    if len(cases):
        valid_case_keys = cases.dropna(
            subset=["run_number", "object_id"]
        )
        case_keys = {
            (int(row.run_number), str(row.object_id))
            for row in valid_case_keys[
                ["run_number", "object_id"]
            ].itertuples(index=False)
        }

    all_realization_keys = visible_all_keys | case_keys
    detected_realization_keys = visible_detected_keys | case_keys
    total_realizations = int(len(all_realization_keys))
    detected_realizations = int(len(detected_realization_keys))

    metrics.extend([
        _metric_row(
            "total_realizations",
            value=total_realizations,
            units="realizations",
        ),
        _metric_row(
            "initially_detected_realizations",
            numerator=detected_realizations,
            denominator=total_realizations,
            units="realizations",
        ),
        _metric_row(
            "detected_spacecraft_cases",
            value=detected_spacecraft_cases,
            units="spacecraft cases",
        ),
    ])

    n_cases = int(len(cases))
    iod_success_cases = int(cases["iod_success"].sum()) if n_cases else 0
    od_started_cases = int(cases["od_started"].sum()) if n_cases else 0
    converged_cases = int(cases["converged"].sum()) if n_cases else 0

    metrics.extend([
        _metric_row(
            "iod_cases",
            value=n_cases,
            units="IOD cases",
        ),
        _metric_row(
            "iod_success_given_iod_case",
            numerator=iod_success_cases,
            denominator=n_cases,
            units="IOD cases",
        ),
        _metric_row(
            "od_started_given_iod_success",
            numerator=od_started_cases,
            denominator=iod_success_cases,
            units="OD cases",
        ),
        _metric_row(
            "converged_given_od_started",
            numerator=converged_cases,
            denominator=od_started_cases,
            units="OD cases",
        ),
    ])

    if n_cases and total_realizations:
        valid_keys = cases.dropna(
            subset=["run_number", "object_id"]
        ).copy()
        valid_keys["run_number"] = valid_keys["run_number"].astype(int)
        realization_outcomes = valid_keys.groupby(
            ["run_number", "object_id"], as_index=False
        ).agg(
            any_converged=("converged", "any"),
            any_iod_success=("iod_success", "any"),
            any_od_started=("od_started", "any"),
        )
        converged_realizations = int(
            realization_outcomes["any_converged"].sum()
        )
        iod_success_realizations = int(
            realization_outcomes["any_iod_success"].sum()
        )
        od_started_realizations = int(
            realization_outcomes["any_od_started"].sum()
        )
    else:
        converged_realizations = 0
        iod_success_realizations = 0
        od_started_realizations = 0

    metrics.extend([
        _metric_row(
            "iod_successful_realizations",
            numerator=iod_success_realizations,
            denominator=detected_realizations,
            units="realizations",
        ),
        _metric_row(
            "od_started_realizations",
            numerator=od_started_realizations,
            denominator=iod_success_realizations,
            units="realizations",
        ),
        _metric_row(
            "converged_all_realizations",
            numerator=converged_realizations,
            denominator=total_realizations,
            units="realizations",
        ),
        _metric_row(
            "converged_given_detected_realization",
            numerator=converged_realizations,
            denominator=detected_realizations,
            units="realizations",
        ),
    ])

    for column, metric_name in (
        ("lost_non_ems", "lost_non_ems_given_od_started"),
        (
            "lost_after_ems_reacquisition",
            "lost_after_ems_reacquisition_given_od_started",
        ),
        ("time_limit", "time_limit_given_od_started"),
        ("step_limit", "step_limit_given_od_started"),
        ("runtime_error", "runtime_error_given_od_started"),
        ("incomplete", "incomplete_given_iod_case"),
    ):
        numerator = int(cases[column].sum()) if n_cases else 0
        denominator = (
            n_cases if column == "incomplete" else od_started_cases
        )
        metrics.append(
            _metric_row(
                metric_name,
                numerator=numerator,
                denominator=denominator,
                units="cases",
            )
        )

    blackout_cases = int(
        cases["ever_ems_blackout"].sum()
    ) if n_cases else 0
    blackout_episodes = int(
        cases["n_ems_blackout_episodes"].sum()
    ) if n_cases else 0
    successful_reacquisitions = int(
        cases["n_successful_reacquisitions"].sum()
    ) if n_cases else 0
    metrics.extend([
        _metric_row(
            "experienced_ems_blackout_given_od_started",
            numerator=blackout_cases,
            denominator=od_started_cases,
            units="OD cases",
        ),
        _metric_row(
            "successful_reacquisition_given_blackout_episode",
            numerator=successful_reacquisitions,
            denominator=blackout_episodes,
            units="blackout episodes",
        ),
        _metric_row(
            "total_ems_blackout_duration_s",
            value=(
                float(cases["total_ems_blackout_duration_s"].sum())
                if n_cases
                else 0.0
            ),
            units="s",
        ),
    ])

    return pd.DataFrame(metrics)


def generate_mission_summary(
    config: Mapping[str, Any],
) -> dict[str, Path]:
    """Generate case-level and campaign-level mission summary CSVs."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")

    top_dir = Path(os.path.abspath(str(config["top_dir"])))
    summary_cfg = config.get("mission_summary", {}) or {}
    output_value = Path(
        str(summary_cfg.get("output_folder", "mission_summary"))
    ).expanduser()
    output_dir = (
        output_value
        if output_value.is_absolute()
        else top_dir / output_value
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    visible_dir = top_dir / str(config["visible_files_folder"])
    master_path = top_dir / "MASTER_IOD.csv"
    od_dir = top_dir / str(config["od_file_dir"])
    od_diag_cfg = config.get("od_diagnostics", {}) or {}
    outer_dir = od_dir / str(
        od_diag_cfg.get("outer_loop_dir", "outer_loop")
    )
    detail_root = od_dir / str(
        od_diag_cfg.get("detail_root_dir", "detailed")
    )

    visible = _load_visible_tables(visible_dir)

    if master_path.is_file():
        master = pd.read_csv(master_path)
    else:
        master = pd.DataFrame()

    case_rows: list[dict[str, Any]] = []
    if len(master):
        run_numbers = _infer_master_run_numbers(
            master,
            number_of_runs=int(config.get("number_of_runs", 1)),
        )
        outer_by_master_row = _index_outer_files(outer_dir)

        for row_index, master_row in master.iterrows():
            outer_path = _choose_outer_path(
                master_row,
                master_row_index=int(row_index),
                top_dir=top_dir,
                outer_by_master_row=outer_by_master_row,
            )
            case_rows.append(
                _case_summary_row(
                    master_row,
                    master_row_index=int(row_index),
                    run_number=_finite_float(
                        run_numbers.at[row_index]
                    ),
                    outer_path=outer_path,
                    detail_root=detail_root,
                )
            )

    case_columns = [
        "master_row_idx",
        "run_number",
        "object_id",
        "tbo_H",
        "detecting_spacecraft_id",
        "index_used",
        "initially_detected",
        "iod_success",
        "od_started",
        "outcome",
        "terminal_event",
        "terminal_reason",
        "converged",
        "lost_non_ems",
        "lost_after_ems_reacquisition",
        "time_limit",
        "step_limit",
        "runtime_error",
        "incomplete",
        "ever_ems_blackout",
        "n_ems_blackout_episodes",
        "total_ems_blackout_duration_s",
        "max_ems_blackout_duration_s",
        "n_reacquisition_attempts",
        "n_successful_reacquisitions",
        "n_prediction_only_cycles",
        "n_od_updates",
        "n_detected_cycles",
        "time_to_convergence_days",
        "final_position_error_km",
        "final_velocity_error_km_s",
        "final_position_cov_trace_km2",
        "final_velocity_cov_trace_km2_s2",
        "final_nis_mean",
        "run_uid",
        "outer_csv_path",
    ]
    cases = pd.DataFrame(case_rows, columns=case_columns)

    statistics = _campaign_statistics(
        visible,
        cases,
        include_ems_exclusion=bool(
            config.get("INCLUDE_EMS_EXCLUSION", False)
        ),
    )

    case_path = output_dir / "mission_case_summary.csv"
    statistics_path = output_dir / "mission_statistics.csv"
    _atomic_to_csv(cases, case_path)
    _atomic_to_csv(statistics, statistics_path)

    print(
        f"[Mission Summary] cases={len(cases)}, "
        f"visible_realizations="
        f"{visible[['run_number', 'object_id']].drop_duplicates().shape[0] if len(visible) else 0}",
        flush=True,
    )
    return {
        "case_summary": case_path.resolve(),
        "statistics": statistics_path.resolve(),
    }


def main() -> None:
    parser = ArgumentParser(
        description="Generate mission outcome summary CSVs."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the main simulation YAML.",
    )
    args = parser.parse_args()

    from simulation_config import load_simulation_config

    config = load_simulation_config(args.config)
    outputs = generate_mission_summary(config)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
