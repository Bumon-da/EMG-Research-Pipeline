"""
Gesture Distribution

Sample count per gesture label for a chosen pipeline run, one chart per
exercise. NinaPro's label numbering resets at 0 for every exercise (label
5 in exercise 1 is a different gesture than label 5 in exercise 2), so
this deliberately never merges exercises into a single chart - see
`src/data/validator.py` for where that distinction is enforced.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard.data_access import load_class_distribution
from src.dashboard.state import run_picker

st.set_page_config(page_title="Gesture Distribution - EMG Pipeline", layout="wide")

st.title("Gesture Distribution")

run = run_picker()

if run is None:
    st.stop()

class_df = load_class_distribution(run)

if class_df.empty:
    st.warning(
        "No class distribution data found for this run "
        "(validation/class_distribution.csv is missing)."
    )
    st.stop()

class_df = class_df.copy()
class_df["Type"] = class_df["Label"].apply(lambda label: "Rest" if label == 0 else "Gesture")

exercises = sorted(class_df["Exercise"].unique())

tabs = st.tabs([f"Exercise {ex}" for ex in exercises])

for tab, exercise in zip(tabs, exercises):
    with tab:
        exercise_df = class_df[class_df["Exercise"] == exercise].sort_values("Label")

        gesture_only = exercise_df[exercise_df["Label"] != 0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Gestures", len(gesture_only))
        col2.metric("Rest samples", f"{int(exercise_df[exercise_df['Label'] == 0]['Samples'].sum()):,}")

        if not gesture_only.empty and gesture_only["Samples"].min() > 0:
            imbalance = gesture_only["Samples"].max() / gesture_only["Samples"].min()
            col3.metric("Imbalance ratio (max/min)", f"{imbalance:.2f}")
        else:
            col3.metric("Imbalance ratio (max/min)", "-")

        fig = px.bar(
            exercise_df,
            x="Label",
            y="Samples",
            color="Type",
            color_discrete_map={"Rest": "#8C8C8C", "Gesture": "#C44E52"},
            labels={"Samples": "Sample Count", "Label": "Gesture Label (0 = rest)"},
        )
        fig.update_xaxes(type="category")
        fig.update_layout(margin=dict(l=40, r=20, t=20, b=40), height=400)

        st.plotly_chart(fig, width="stretch")

        with st.expander("Data"):
            st.dataframe(exercise_df, width="stretch", hide_index=True)
