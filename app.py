"""
EMG Research Pipeline Dashboard - Home

Run with:
    streamlit run app.py

Overview of the dataset on disk and past pipeline runs, plus a button to
run the full pipeline (load -> validate -> EDA) and generate a new run.
Other pages (sidebar) explore a run's results in more detail.
"""

from __future__ import annotations

import streamlit as st

from config.settings import RAW_DATA_DIR
from src.dashboard.data_access import (
    dataset_root_exists,
    list_experiment_runs,
    list_subjects_on_disk,
    list_trials_on_disk,
    load_manifest,
    run_full_pipeline,
)
from src.dashboard.state import set_selected_run_name

st.set_page_config(page_title="EMG Research Pipeline", layout="wide")

st.title("EMG Research Pipeline Dashboard")
st.caption(
    "Explore the NinaPro DB1 dataset, dataset validation results, gesture "
    "class balance, and cross-subject comparisons for this project."
)

# ----------------------------------------------------------------------
# Dataset on disk
# ----------------------------------------------------------------------

st.header("Dataset")

if not dataset_root_exists():
    st.error(
        f"Raw dataset directory not found: `{RAW_DATA_DIR}`. Set the "
        "`EMG_RAW_DATA_DIR` environment variable, or check "
        "config/settings.py."
    )
    subjects: list[str] = []
else:
    subjects = list_subjects_on_disk()
    st.caption(f"Dataset root: `{RAW_DATA_DIR}`")

    col1, col2 = st.columns(2)
    col1.metric("Subjects on disk", len(subjects))

    if subjects:
        total_trials = sum(len(list_trials_on_disk(s)) for s in subjects)
        col2.metric("Trials on disk", total_trials)

        with st.expander("Subjects"):
            st.write(", ".join(subjects))
    else:
        st.warning("No subject folders found under the raw dataset directory.")

st.divider()

# ----------------------------------------------------------------------
# Pipeline runs
# ----------------------------------------------------------------------

st.header("Pipeline Runs")

st.write(
    "Running the pipeline loads the full dataset, runs validation, and "
    "generates EDA figures/reports into a new, isolated output folder. "
    "This can take a few minutes for the full 27-subject dataset."
)

if st.button("Run Full Pipeline Now", type="primary", disabled=not dataset_root_exists()):
    status_box = st.status("Running pipeline...", expanded=True)

    def report(message: str) -> None:
        status_box.write(message)

    try:
        result = run_full_pipeline(status_callback=report)
        status_box.update(label="Pipeline complete.", state="complete")
        set_selected_run_name(result.experiment.experiment_name)
        st.success(f"New run created: {result.experiment.experiment_name}")
        st.rerun()
    except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
        status_box.update(label="Pipeline failed.", state="error")
        st.exception(exc)

runs = list_experiment_runs()

if not runs:
    st.info(
        "No pipeline runs yet. Click 'Run Full Pipeline Now' above, then "
        "use the pages in the sidebar to explore the results."
    )
else:
    st.write(f"{len(runs)} run(s) found. Use the pages in the sidebar to explore a run's results.")

    rows = []
    for run in runs:
        manifest = load_manifest(run)
        dataset_info = manifest.get("dataset", {})
        rows.append(
            {
                "Run": run.name,
                "Created": manifest.get("created_at", ""),
                "Subjects": dataset_info.get("subjects", ""),
                "Trials": dataset_info.get("trials", ""),
                "Flagged Trials": dataset_info.get("flagged_trials", ""),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)

st.divider()
st.caption(
    "Pages: **Raw Signal Browser** (browse individual trials), "
    "**Validation** (integrity checks for a run), **Gesture "
    "Distribution** (class balance per exercise), **Cross-Subject "
    "Comparison** (signal quality across all subjects)."
)
