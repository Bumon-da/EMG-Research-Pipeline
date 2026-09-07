"""
Features

Surfaces the v0.6.0 feature-extraction stage (Atzori et al. 2014 NinaPro
DB1 baseline: RMS, MAV, WL, IEMG, MCR, SSC, HIST, mDWT) for a chosen
pipeline run, without loading the full per-window feature matrix into the
dashboard process.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import streamlit as st

from src.dashboard.data_access import (
    load_feature_manifest,
    load_feature_normalization_stats,
    load_feature_summary,
    load_features_report,
)
from src.dashboard.state import run_picker
from src.features.feature_matrix import list_subject_feature_files

st.set_page_config(page_title="Features - EMG Pipeline", layout="wide")

st.title("Features")

run = run_picker()

if run is None:
    st.stop()

summary = load_feature_summary(run)
manifest_df = load_feature_manifest(run)
normalization_df = load_feature_normalization_stats(run)
report_text = load_features_report(run)

if not summary:
    st.warning(
        "No feature output found for this run "
        "(features/feature_summary.json is missing). The run may have "
        "failed before feature extraction completed, or predates v0.6.0."
    )
    st.stop()

# ----------------------------------------------------------------------
# Headline metrics
# ----------------------------------------------------------------------

row1 = st.columns(4)
row1[0].metric("Subjects Processed", summary.get("subjects_processed", "-"))
row1[1].metric("Feature Columns", summary.get("feature_columns", "-"))
row1[2].metric("Windows Kept (Train)", f"{summary.get('total_windows_train', 0):,}")
row1[3].metric("Windows Kept (Test)", f"{summary.get('total_windows_test', 0):,}")

row2 = st.columns(4)
row2[0].metric("Wavelet", f"{summary.get('wavelet', '-')} (L{summary.get('wavelet_level', '-')})")
row2[1].metric("Histogram Bins", summary.get("hist_bins", "-"))
row2[2].metric("Test-Fallback Subjects", summary.get("subjects_with_train_fallback", "-"))
row2[3].metric("Elapsed", f"{summary.get('elapsed_seconds', '-')}s")

st.caption(
    "Frequency-domain features (MDF/MNF/PSD) are deliberately not "
    "included - at a 20-sample window / 100Hz, an FFT gives ~5Hz bins "
    "over an already-rectified sensor envelope, not a motor-unit firing "
    "spectrum. Train classifiers on `GlobalLabel`, not `Label` - NinaPro "
    "DB1 restarts gesture numbering at 1 for every exercise file."
)

st.divider()

# ----------------------------------------------------------------------
# Feature columns by group
# ----------------------------------------------------------------------

st.subheader("Feature Columns")

if manifest_df.empty:
    st.info("No feature manifest found for this run.")
else:
    by_group = manifest_df.groupby("Group").size().reset_index(name="Columns")
    cols = st.columns(len(by_group))
    for col, (_, row) in zip(cols, by_group.iterrows()):
        col.metric(str(row["Group"]).replace("_", " ").title(), int(row["Columns"]))

    with st.expander(f"Full feature manifest ({len(manifest_df)} columns)"):
        st.dataframe(manifest_df, width="stretch", hide_index=True)

st.divider()

# ----------------------------------------------------------------------
# Per-subject row counts (cheap - Parquet metadata only, no data read)
# ----------------------------------------------------------------------

st.subheader("Per-Subject Feature Files")

feature_files = list_subject_feature_files(run.features_dir)

if not feature_files:
    st.info("No per-subject feature files found for this run.")
else:
    rows = []
    for filepath in feature_files:
        metadata = pq.ParquetFile(filepath).metadata
        rows.append(
            {
                "Subject": filepath.stem.replace("_features", ""),
                "Windows": metadata.num_rows,
                "SizeMB": round(filepath.stat().st_size / (1024 * 1024), 2),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

st.divider()

# ----------------------------------------------------------------------
# Normalization stats + report
# ----------------------------------------------------------------------

with st.expander("Normalization statistics (recomputed for this run's split)"):
    if normalization_df.empty:
        st.info("No normalization statistics found for this run.")
    else:
        st.dataframe(normalization_df, width="stretch", hide_index=True)

with st.expander("Feature extraction report (text)"):
    st.text(report_text if report_text else "No feature extraction report found for this run.")
