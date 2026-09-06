"""
EDA Report

Builds a short, human-readable text summary of an EDA run, meant to sit
alongside the CSV/figure outputs so a reader can see the headline numbers
without opening a spreadsheet.
"""

from __future__ import annotations

import pandas as pd


def build_eda_report(
    total_subjects: int,
    total_trials: int,
    total_samples: int,
    sampled_trial_filenames: list[str],
    summary_df: pd.DataFrame,
) -> str:

    lines = [
        "EDA SUMMARY REPORT",
        "=" * 60,
        f"Subjects : {total_subjects}",
        f"Trials   : {total_trials}",
        f"Samples  : {total_samples:,}",
        "",
        "Per-trial descriptive statistics were computed for every trial "
        "(see per-subject *_statistics.csv files).",
        "",
        "Detailed figures (raw signal, histograms, boxplots, channel "
        "correlation) were generated for a sample of trials rather than "
        "all of them, to keep output size manageable:",
    ]

    if sampled_trial_filenames:
        for filename in sampled_trial_filenames:
            lines.append(f"  - {filename}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(
        "Dataset-wide figures (subject_comparison.png, "
        "gesture_distribution.png) were generated across all subjects."
    )
    lines.append("")
    lines.append("Trials by subject:")

    if not summary_df.empty:
        per_subject = summary_df.groupby("Subject").size()
        for subject, count in per_subject.items():
            lines.append(f"  {subject:<6}: {count} trial(s)")

    return "\n".join(lines)
