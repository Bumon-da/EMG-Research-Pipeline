"""
Validation

Surfaces the dataset integrity checks (missing/NaN/negative values,
saturation, duplicate trials, duration outliers, low-activity channels)
for a chosen pipeline run, without having to open the raw CSVs.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.data_access import (
    load_flagged_trials,
    load_integrity_report,
    load_validation_details,
    load_validation_summary,
)
from src.dashboard.state import run_picker

st.set_page_config(page_title="Validation - EMG Pipeline", layout="wide")

st.title("Validation")

run = run_picker()

if run is None:
    st.stop()

summary_df = load_validation_summary(run)
details_df = load_validation_details(run)
flagged_df = load_flagged_trials(run)
report_text = load_integrity_report(run)

if summary_df.empty:
    st.warning(
        "No validation output found for this run (validation_summary.csv "
        "is missing). The run may have failed before validation completed."
    )
    st.stop()


def _metric(name: str, default: str = "-") -> str:
    row = summary_df[summary_df["Metric"] == name]
    if row.empty:
        return default

    value = row["Value"].iloc[0]

    # validation_summary.csv's Value column mixes ints/floats/NaN, so
    # pandas reads the whole column as float - show whole numbers without
    # a trailing ".0" for readability.
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.2f}"
    except (TypeError, ValueError):
        return str(value)


# ----------------------------------------------------------------------
# Headline metrics
# ----------------------------------------------------------------------

row1 = st.columns(4)
row1[0].metric("Subjects", _metric("Subjects"))
row1[1].metric("Trials", _metric("Trials"))
row1[2].metric("Flagged Trials", _metric("Flagged Trials"))
row1[3].metric("Duplicate Trials", _metric("Duplicate Trials"))

row2 = st.columns(4)
row2[0].metric("Saturated Channels", _metric("Saturated Channels"))
row2[1].metric("Low-Activity Channels", _metric("Low-Activity Channels"))
row2[2].metric("Duration Outliers", _metric("Duration Outliers"))
row2[3].metric(
    "Missing Refined Labels", _metric("Trials Missing Refined Labels")
)

st.divider()

# ----------------------------------------------------------------------
# Flagged trials
# ----------------------------------------------------------------------

st.subheader("Flagged Trials")

if flagged_df.empty:
    st.success("No trials were flagged by validation.")
else:
    subject_options = ["All"] + sorted(flagged_df["subject"].unique().tolist())
    selected_subject = st.selectbox("Filter by subject", subject_options)

    view_df = (
        flagged_df
        if selected_subject == "All"
        else flagged_df[flagged_df["subject"] == selected_subject]
    )

    st.dataframe(view_df, width="stretch", hide_index=True)

st.divider()

# ----------------------------------------------------------------------
# Full details + report
# ----------------------------------------------------------------------

with st.expander(f"All validation details ({len(details_df)} trials)"):
    st.dataframe(details_df, width="stretch", hide_index=True)

with st.expander("Integrity report (text)"):
    st.text(report_text if report_text else "No integrity report found for this run.")
