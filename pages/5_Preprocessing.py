"""
Preprocessing

Surfaces the v0.5.0 signal-preprocessing stage (rectification guard,
per-subject/per-channel train-split-only z-score normalization, and
label/split-aware windowing) for a chosen pipeline run, without having to
open the raw CSVs.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard.data_access import (
    load_normalization_stats,
    load_preprocessing_report,
    load_preprocessing_summary,
    load_rectification_report,
    load_window_drop_summary,
    load_window_tally,
)
from src.dashboard.state import run_picker

st.set_page_config(page_title="Preprocessing - EMG Pipeline", layout="wide")

st.title("Preprocessing")

run = run_picker()

if run is None:
    st.stop()

summary = load_preprocessing_summary(run)
normalization_df = load_normalization_stats(run)
rectification_df = load_rectification_report(run)
window_tally_df = load_window_tally(run)
window_drop_df = load_window_drop_summary(run)
report_text = load_preprocessing_report(run)

if not summary:
    st.warning(
        "No preprocessing output found for this run "
        "(preprocessing/preprocessing_summary.json is missing). The run "
        "may have failed before preprocessing completed."
    )
    st.stop()

# ----------------------------------------------------------------------
# Headline metrics
# ----------------------------------------------------------------------

row1 = st.columns(4)
row1[0].metric("Subjects Processed", summary.get("subjects_processed", "-"))
row1[1].metric("Subjects Skipped", summary.get("subjects_with_zero_trials", "-"))
row1[2].metric("Test-Fallback Subjects", summary.get("subjects_with_train_fallback", "-"))
row1[3].metric("Non-Finite Samples Found", f"{summary.get('total_non_finite_samples_found', 0):,}")

row2 = st.columns(4)
row2[0].metric("Windows Kept (Train)", f"{summary.get('total_windows_train', 0):,}")
row2[1].metric("Windows Kept (Test)", f"{summary.get('total_windows_test', 0):,}")
row2[2].metric("Dropped (Transition)", f"{summary.get('total_windows_dropped_transition', 0):,}")
row2[3].metric("Dropped (Boundary)", f"{summary.get('total_windows_dropped_boundary', 0):,}")

st.caption(
    "No bandpass/notch filter is applied to this dataset - NinaPro DB1's "
    "`emg` field is already a rectified, non-negative sensor envelope, "
    "not raw sEMG. `src/preprocessing/filtering.py` implements a real "
    "Butterworth bandpass + notch filter for a future raw-sEMG source; "
    "it is not invoked here."
)

st.divider()

# ----------------------------------------------------------------------
# Per-subject normalization stats
# ----------------------------------------------------------------------

st.subheader("Per-Subject Normalization Statistics")
st.caption(
    "Per-channel mean/std, computed from each subject's TRAIN-split "
    "samples only (see `src/preprocessing/normalization.py`) and applied "
    "to both that subject's train and test rows."
)

if normalization_df.empty:
    st.info("No normalization statistics found for this run.")
else:
    subject_options = ["All"] + sorted(normalization_df["Subject"].unique().tolist())
    selected_subject = st.selectbox("Filter by subject", subject_options)

    view_df = (
        normalization_df
        if selected_subject == "All"
        else normalization_df[normalization_df["Subject"] == selected_subject]
    )

    st.dataframe(view_df, width="stretch", hide_index=True)

    flagged_fallback = normalization_df[normalization_df["UsedTestFallback"]]
    if not flagged_fallback.empty:
        st.warning(
            f"{flagged_fallback['Subject'].nunique()} subject(s) had no "
            f"train-split samples and fell back to their full data for "
            f"normalization statistics (see `UsedTestFallback`)."
        )

st.divider()

# ----------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------

st.subheader("Window Counts by Exercise")
st.caption("Windows kept per (exercise, label, split); label 0 = rest.")

if window_tally_df.empty:
    st.info("No window tally found for this run.")
else:
    by_exercise_split = (
        window_tally_df.groupby(["Exercise", "Split"])["WindowCount"].sum().reset_index()
    )

    fig = px.bar(
        by_exercise_split,
        x="Exercise",
        y="WindowCount",
        color="Split",
        barmode="group",
        labels={"WindowCount": "Windows Kept", "Exercise": "Exercise"},
    )
    fig.update_xaxes(type="category")
    fig.update_layout(margin=dict(l=40, r=20, t=20, b=40), height=400)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Full window tally"):
        st.dataframe(window_tally_df, width="stretch", hide_index=True)

st.divider()

# ----------------------------------------------------------------------
# Drop reasons + rectification
# ----------------------------------------------------------------------

st.subheader("Window Drop Reasons")
st.caption(
    "A candidate window is dropped if its label is not uniform (a "
    "gesture transition mid-window) or if it straddles the train/test "
    "split boundary."
)

if window_drop_df.empty:
    st.info("No window drop summary found for this run.")
else:
    totals = window_drop_df[["DroppedTransition", "DroppedBoundary"]].sum()
    drop_fig = px.bar(
        x=["Dropped: Transition", "Dropped: Boundary"],
        y=[int(totals["DroppedTransition"]), int(totals["DroppedBoundary"])],
        labels={"x": "Reason", "y": "Windows Dropped"},
    )
    drop_fig.update_layout(margin=dict(l=40, r=20, t=20, b=40), height=350)
    st.plotly_chart(drop_fig, width="stretch")

    with st.expander(f"Per-trial drop summary ({len(window_drop_df)} trials)"):
        st.dataframe(window_drop_df, width="stretch", hide_index=True)

with st.expander("Rectification report (negative/non-finite samples per trial)"):
    if rectification_df.empty:
        st.info("No rectification report found for this run.")
    else:
        st.dataframe(rectification_df, width="stretch", hide_index=True)

with st.expander("Preprocessing report (text)"):
    st.text(report_text if report_text else "No preprocessing report found for this run.")
