"""
Preprocessing Report

Builds a short, human-readable text summary of a signal-preprocessing
run, mirroring src/eda/report.py's role: sits alongside the CSV/JSON
outputs so headline numbers are readable without opening a spreadsheet.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_preprocessing_report(
    normalization_stats_df: pd.DataFrame,
    rectification_df: pd.DataFrame,
    window_tally_df: pd.DataFrame,
    window_drop_summary_df: pd.DataFrame,
    summary: dict[str, Any],
) -> str:

    lines = [
        "SIGNAL PREPROCESSING REPORT",
        "=" * 60,
        f"Subjects processed            : {summary.get('subjects_processed', 0)}",
        f"Subjects skipped (no trials)  : {summary.get('subjects_with_zero_trials', 0)}",
        f"Subjects using test fallback  : {summary.get('subjects_with_train_fallback', 0)}",
        "",
        "Filtering: no bandpass/notch filter is applied to this dataset.",
        "NinaPro DB1's emg field is already a rectified, non-negative sensor",
        "envelope (see src/data/validator.py) - a bandpass/notch filter tuned",
        "for raw sEMG would risk distorting it. src/preprocessing/filtering.py",
        "implements one for future raw-sEMG sources; it is not invoked here.",
        "",
        f"Negative samples clipped       : {summary.get('total_negative_samples_clipped', 0):,}",
        f"Non-finite samples found       : {summary.get('total_non_finite_samples_found', 0):,}",
        "",
        "Normalization: per-subject, per-channel z-score, computed from",
        "each subject's TRAIN-split samples only and applied to both that",
        "subject's train and test rows (see normalization_stats.csv).",
    ]

    zero_std_count = 0
    if not normalization_stats_df.empty and "ZeroStdFlagged" in normalization_stats_df:
        zero_std_count = int(normalization_stats_df["ZeroStdFlagged"].sum())
    lines.append(f"Zero-variance channels floored : {zero_std_count}")

    lines.append("")
    lines.append(
        f"Windowing: {summary.get('total_windows', 0):,} total windows kept "
        f"({summary.get('total_windows_train', 0):,} train / "
        f"{summary.get('total_windows_test', 0):,} test)."
    )
    lines.append(
        f"Dropped: {summary.get('total_windows_dropped_transition', 0):,} for a label "
        f"transition, {summary.get('total_windows_dropped_boundary', 0):,} for straddling "
        f"the train/test split boundary."
    )

    if not window_tally_df.empty:
        lines.append("")
        lines.append("Window counts by exercise (train+test, label 0 = rest):")
        by_exercise = window_tally_df.groupby("Exercise")["WindowCount"].sum().sort_index()
        for exercise, count in by_exercise.items():
            lines.append(f"  Exercise {exercise}: {int(count):,} windows")

    return "\n".join(lines)
