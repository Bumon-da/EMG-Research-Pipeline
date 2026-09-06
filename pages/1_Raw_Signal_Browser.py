"""
Raw Signal Browser

Pick a subject and trial, then explore the raw EMG (and glove, if present)
signal interactively: choose channels, zoom into a time window, and see
the gesture-label track lined up underneath the signal.

Loads exactly one `.mat` file on demand - never the whole dataset.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.settings import SAMPLING_RATE
from src.dashboard.data_access import (
    dataset_root_exists,
    list_subjects_on_disk,
    list_trials_on_disk,
    load_single_trial,
)
from src.eda.statistics import Statistics

st.set_page_config(page_title="Raw Signal Browser - EMG Pipeline", layout="wide")

st.title("Raw Signal Browser")

if not dataset_root_exists():
    st.error("Raw dataset directory not found. Check the Home page for details.")
    st.stop()

subjects = list_subjects_on_disk()

if not subjects:
    st.warning("No subjects found on disk.")
    st.stop()


@st.cache_data(show_spinner="Loading trial...")
def _load_trial_cached(subject_id: str, filename: str):
    trial = load_single_trial(subject_id, filename)
    if trial is None:
        return None
    # Cache-friendly plain data - Trial itself is fine to cache too, but
    # returning it directly keeps this simple since it's just numpy arrays.
    return trial


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------

col_subject, col_trial = st.columns(2)

with col_subject:
    subject_id = st.selectbox("Subject", subjects)

trial_filenames = list_trials_on_disk(subject_id)

if not trial_filenames:
    st.warning(f"No trial files found for subject {subject_id}.")
    st.stop()

with col_trial:
    filename = st.selectbox("Trial", trial_filenames)

trial = _load_trial_cached(subject_id, filename)

if trial is None:
    st.error(f"Could not load {subject_id}/{filename} - see the app log for details.")
    st.stop()

duration_sec = trial.samples / SAMPLING_RATE

# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------

meta_cols = st.columns(6)
meta_cols[0].metric("Samples", f"{trial.samples:,}")
meta_cols[1].metric("Channels", trial.channels)
meta_cols[2].metric("Duration (s)", f"{duration_sec:.1f}")
meta_cols[3].metric("Exercise", trial.exercise_id if trial.exercise_id is not None else "-")
meta_cols[4].metric("Refined labels", "Yes" if trial.has_refined_labels else "No")
meta_cols[5].metric("Glove data", "Yes" if trial.has_glove else "No")

st.divider()

# ----------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------

st.subheader("Signal")

control_cols = st.columns([2, 3])

with control_cols[0]:
    channel_labels = [f"Ch{i + 1}" for i in range(trial.channels)]
    default_channels = channel_labels[: min(3, trial.channels)]
    selected_channels = st.multiselect(
        "Channels", channel_labels, default=default_channels
    )

with control_cols[1]:
    default_window = (0.0, float(min(10.0, duration_sec)))
    time_range = st.slider(
        "Time window (s)",
        min_value=0.0,
        max_value=float(duration_sec),
        value=default_window,
        step=0.5 if duration_sec > 20 else 0.1,
    )

if not selected_channels:
    st.info("Select at least one channel to plot.")
    st.stop()

start_idx = int(time_range[0] * SAMPLING_RATE)
end_idx = max(start_idx + 1, int(time_range[1] * SAMPLING_RATE))
end_idx = min(end_idx, trial.samples)

n_points = end_idx - start_idx
if n_points > 200_000:
    st.warning(
        f"Selected window has {n_points:,} samples - narrowing the time "
        "window will make the plot more responsive."
    )

time_axis = np.arange(start_idx, end_idx) / SAMPLING_RATE
labels = np.asarray(trial.labels).reshape(-1)[start_idx:end_idx]

# ----------------------------------------------------------------------
# Signal + label plot
# ----------------------------------------------------------------------

fig = make_subplots(
    rows=len(selected_channels) + 1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[3] * len(selected_channels) + [1],
    subplot_titles=selected_channels + ["Gesture label"],
)

for row, channel_label in enumerate(selected_channels, start=1):
    ch_index = int(channel_label.replace("Ch", "")) - 1
    fig.add_trace(
        go.Scattergl(
            x=time_axis,
            y=trial.emg[start_idx:end_idx, ch_index],
            mode="lines",
            name=channel_label,
            line=dict(width=1),
            showlegend=False,
        ),
        row=row,
        col=1,
    )

fig.add_trace(
    go.Scattergl(
        x=time_axis,
        y=labels,
        mode="lines",
        line=dict(width=1, shape="hv", color="#C44E52"),
        name="Label",
        showlegend=False,
    ),
    row=len(selected_channels) + 1,
    col=1,
)

fig.update_xaxes(title_text="Time (s)", row=len(selected_channels) + 1, col=1)
fig.update_layout(
    height=180 * (len(selected_channels) + 1),
    margin=dict(l=40, r=20, t=30, b=40),
)

st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------
# Glove data (optional)
# ----------------------------------------------------------------------

if trial.has_glove:
    with st.expander("Glove (joint-angle) data"):
        n_glove_channels = trial.glove.shape[1]
        glove_labels = [f"Glove{i + 1}" for i in range(n_glove_channels)]
        default_glove = glove_labels[: min(5, n_glove_channels)]
        selected_glove = st.multiselect(
            "Glove channels", glove_labels, default=default_glove, key="glove_channels"
        )

        if selected_glove:
            glove_fig = go.Figure()
            for glove_label in selected_glove:
                idx = int(glove_label.replace("Glove", "")) - 1
                glove_fig.add_trace(
                    go.Scattergl(
                        x=time_axis,
                        y=trial.glove[start_idx:end_idx, idx],
                        mode="lines",
                        name=glove_label,
                        line=dict(width=1),
                    )
                )
            glove_fig.update_layout(
                height=300,
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis_title="Time (s)",
                yaxis_title="Joint angle (a.u.)",
            )
            st.plotly_chart(glove_fig, width="stretch")

st.divider()

# ----------------------------------------------------------------------
# Descriptive statistics
# ----------------------------------------------------------------------

st.subheader("Descriptive Statistics (full trial)")
stats_df = Statistics.descriptive_statistics(trial.emg)
st.dataframe(stats_df, width="stretch")
