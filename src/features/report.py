"""
Feature Extraction Report

Builds a short, human-readable text summary of a feature-extraction run,
mirroring src/preprocessing/report.py's role: sits alongside the
Parquet/CSV/JSON outputs so headline numbers and the deliberate
deviations from Atzori et al. 2014 are readable without opening a
spreadsheet.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_features_report(summary: dict[str, Any], feature_manifest_df: pd.DataFrame) -> str:

    lines = [
        "FEATURE EXTRACTION REPORT",
        "=" * 60,
        f"Subjects processed             : {summary.get('subjects_processed', 0)}",
        f"Subjects skipped (no trials)    : {summary.get('subjects_with_zero_trials', 0)}",
        f"Subjects using test fallback    : {summary.get('subjects_with_train_fallback', 0)}",
        f"Subjects with no usable windows : {summary.get('subjects_with_no_windows', 0)}",
        "",
        f"Windows kept (train / test)     : "
        f"{summary.get('total_windows_train', 0):,} / {summary.get('total_windows_test', 0):,}",
        f"Feature columns per row         : {summary.get('feature_columns', 0)}",
        f"Elapsed                         : {summary.get('elapsed_seconds', 0)}s",
        "",
        "Feature set: Atzori et al. 2014 NinaPro DB1 baseline (RMS, MAV, WL,",
        "IEMG, MCR, SSC, HIST, mDWT), computed from the per-subject",
        "z-score-normalized signal (never the raw envelope). Frequency-domain",
        "features (MDF/MNF/PSD) are deliberately NOT included: at a 20-sample",
        "window and 100Hz sampling rate, an FFT gives ~5Hz bins over an",
        "already-rectified sensor envelope (see src/data/validator.py), not a",
        "motor-unit firing spectrum - values computed here would not be",
        "comparable to published sEMG frequency-domain results.",
        "",
        "MCR, not ZC: DB1's emg is non-negative, so literature zero-crossing",
        "rate is identically 0 on the raw signal, and even after per-subject",
        "z-scoring, a single 200ms window frequently never crosses the",
        "subject's global mean (measured: 18% of windows all-zero across",
        "every channel). MCR instead counts crossings of each WINDOW's own",
        "mean - not comparable to literature ZC on raw sEMG.",
        "",
        f"Wavelet: {summary.get('wavelet', '?')}, level {summary.get('wavelet_level', '?')} "
        f"(not Atzori's db7 - db7's 14-tap filter cannot decompose a",
        "20-sample window at all; db2 is the shortest common wavelet that",
        "still permits a useful 2-level decomposition).",
        "",
        f"Histogram: {summary.get('hist_bins', 0)} bins with fixed, train-derived-scale",
        "edges (not Atzori's 20 bins - at 20 samples, 20 bins leaves ~91% of",
        "columns structurally zero). Not comparable to HIST20 in the",
        "literature.",
        "",
        "IEMG is exactly window_len x MAV on this fixed-length window -",
        "perfectly collinear. Kept for literature fidelity; drop one before",
        "fitting any covariance-based model (e.g. LDA).",
        "",
        "GlobalLabel: NinaPro DB1 restarts gesture numbering at 1 for every",
        "exercise file. A per-subject feature file spans all three exercises,",
        "so GlobalLabel (Label + a per-exercise cumulative offset) is the",
        "column to train a classifier on - Label alone silently collides",
        "gestures across exercises.",
    ]

    if not feature_manifest_df.empty and "Group" in feature_manifest_df:
        lines.append("")
        lines.append("Columns by group:")
        by_group = feature_manifest_df.groupby("Group").size().sort_index()
        for group, count in by_group.items():
            lines.append(f"  {group}: {int(count)}")

    return "\n".join(lines)
