"""
Shared Streamlit session-state helpers, so every dashboard page agrees on
which experiment run is "current" without re-deriving it independently.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.data_access import ExperimentRun, list_experiment_runs

_STATE_KEY = "emg_dashboard_selected_run"


def get_selected_run_name() -> str | None:
    return st.session_state.get(_STATE_KEY)


def set_selected_run_name(name: str) -> None:
    st.session_state[_STATE_KEY] = name


def run_picker(label: str = "Experiment run") -> ExperimentRun | None:
    """
    Renders a selectbox of available experiment runs (newest first) and
    returns the chosen one, keeping the choice in sync across pages via
    session state. Returns None (after showing a hint) if no runs exist.
    """

    runs = list_experiment_runs()

    if not runs:
        st.info(
            "No pipeline runs found yet. Go to the Home page and click "
            "'Run Full Pipeline Now' to generate validation and EDA output."
        )
        return None

    names = [r.name for r in runs]
    current = get_selected_run_name()
    index = names.index(current) if current in names else 0

    chosen_name = st.selectbox(label, names, index=index)
    set_selected_run_name(chosen_name)

    return next(r for r in runs if r.name == chosen_name)
