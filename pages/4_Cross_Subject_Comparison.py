"""
Cross-Subject Comparison

Compares signal amplitude and data-quality issues across all subjects in
a chosen pipeline run, built entirely from validation_details.csv - no
raw `.mat` files are reloaded here.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard.data_access import load_validation_details
from src.dashboard.state import run_picker

st.set_page_config(page_title="Cross-Subject Comparison - EMG Pipeline", layout="wide")

st.title("Cross-Subject Comparison")

run = run_picker()

if run is None:
    st.stop()

details_df = load_validation_details(run)

if details_df.empty:
    st.warning(
        "No validation details found for this run "
        "(validation/validation_details.csv is missing)."
    )
    st.stop()

# ----------------------------------------------------------------------
# Per-subject summary
# ----------------------------------------------------------------------

issue_columns = [
    "saturated_channels",
    "low_activity_channels",
    "is_duplicate",
    "duration_outlier",
]

per_subject = (
    details_df.groupby("subject")
    .agg(
        trials=("trial", "count"),
        mean_rms=("mean_rms", "mean"),
        flagged_trials=("flagged", "sum"),
        saturated_channels=("saturated_channels", "sum"),
        low_activity_channels=("low_activity_channels", "sum"),
        duplicate_trials=("is_duplicate", "sum"),
        duration_outliers=("duration_outlier", "sum"),
    )
    .reset_index()
    .sort_values("subject")
)

st.subheader("Signal Amplitude by Subject")
st.caption(
    "Mean RMS, averaged across every channel and trial for that subject. "
    "Subjects that stand out from the rest of the cohort may indicate "
    "electrode placement or contact differences worth a closer look."
)

fig = px.bar(
    per_subject,
    x="subject",
    y="mean_rms",
    labels={"subject": "Subject", "mean_rms": "Mean RMS"},
)
fig.update_layout(margin=dict(l=40, r=20, t=20, b=40), height=400)
st.plotly_chart(fig, width="stretch")

st.divider()

st.subheader("Data-Quality Issues by Subject")

melted = per_subject.melt(
    id_vars="subject",
    value_vars=["saturated_channels", "low_activity_channels", "duplicate_trials", "duration_outliers"],
    var_name="Issue",
    value_name="Count",
)

issues_fig = px.bar(
    melted,
    x="subject",
    y="Count",
    color="Issue",
    barmode="group",
    labels={"subject": "Subject"},
)
issues_fig.update_layout(margin=dict(l=40, r=20, t=20, b=40), height=400)
st.plotly_chart(issues_fig, width="stretch")

st.divider()

st.subheader("Per-Subject Table")
st.dataframe(per_subject, width="stretch", hide_index=True)
