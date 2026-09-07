"""
Dashboard Data Access

Kept deliberately separate from the pipeline's own managers: pages read
already-computed CSVs from an experiment run wherever possible (fast, no
dataset reload), and only touch raw `.mat` files for on-demand,
single-trial loads (the raw signal browser). Listing subjects/trials on
disk never parses a `.mat` file - it is plain directory listing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from config.settings import EXPERIMENTS_DIR, RAW_DATA_DIR
from src.data.datamodels import Trial
from src.data.loader import DatasetLoader
from src.pipeline import PipelineResult, run_pipeline


@dataclass
class ExperimentRun:
    name: str
    path: Path

    @property
    def validation_dir(self) -> Path:
        return self.path / "validation"

    @property
    def eda_dir(self) -> Path:
        return self.path / "eda"

    @property
    def preprocessing_dir(self) -> Path:
        return self.path / "preprocessing"

    @property
    def features_dir(self) -> Path:
        return self.path / "features"


# ----------------------------------------------------------------------
# Experiment run discovery + cached outputs
# ----------------------------------------------------------------------


def list_experiment_runs() -> list[ExperimentRun]:
    """
    All experiment runs found under `output/experiments/`, newest first
    (run folder names are timestamp-sortable: `experiment_YYYYMMDD_HHMMSS`).
    """

    root = Path(EXPERIMENTS_DIR)

    if not root.exists():
        return []

    runs = [
        ExperimentRun(name=p.name, path=p) for p in root.iterdir() if p.is_dir()
    ]

    return sorted(runs, key=lambda r: r.name, reverse=True)


def load_manifest(run: ExperimentRun) -> dict:
    manifest_path = run.path / "manifest.json"

    if not manifest_path.exists():
        return {}

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def load_validation_details(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.validation_dir / "validation_details.csv")


def load_flagged_trials(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.validation_dir / "flagged_trials.csv")


def load_validation_summary(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.validation_dir / "validation_summary.csv")


def load_class_distribution(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.validation_dir / "class_distribution.csv")


def load_dataset_summary(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.eda_dir / "dataset_summary.csv")


def load_normalization_stats(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.preprocessing_dir / "normalization_stats.csv")


def load_window_tally(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.preprocessing_dir / "window_tally.csv")


def load_window_drop_summary(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.preprocessing_dir / "window_drop_summary.csv")


def load_rectification_report(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.preprocessing_dir / "rectification_report.csv")


def load_preprocessing_summary(run: ExperimentRun) -> dict:
    summary_path = run.preprocessing_dir / "preprocessing_summary.json"

    if not summary_path.exists():
        return {}

    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_preprocessing_report(run: ExperimentRun) -> str:
    report_path = run.preprocessing_dir / "preprocessing_report.txt"

    if not report_path.exists():
        return ""

    return report_path.read_text(encoding="utf-8")


def load_feature_summary(run: ExperimentRun) -> dict:
    summary_path = run.features_dir / "feature_summary.json"

    if not summary_path.exists():
        return {}

    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_feature_manifest(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.features_dir / "feature_manifest.csv")


def load_feature_normalization_stats(run: ExperimentRun) -> pd.DataFrame:
    return _read_csv_if_exists(run.features_dir / "normalization_stats.csv")


def load_features_report(run: ExperimentRun) -> str:
    report_path = run.features_dir / "features_report.txt"

    if not report_path.exists():
        return ""

    return report_path.read_text(encoding="utf-8")


def load_integrity_report(run: ExperimentRun) -> str:
    report_path = run.validation_dir / "integrity_report.txt"

    if not report_path.exists():
        return ""

    return report_path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Raw dataset access (on-disk listing + on-demand single-trial load)
# ----------------------------------------------------------------------


def dataset_root_exists() -> bool:
    return Path(RAW_DATA_DIR).exists()


def list_subjects_on_disk() -> list[str]:
    """Cheap directory listing - does not parse any `.mat` file."""
    return DatasetLoader().list_subject_ids()


def list_trials_on_disk(subject_id: str) -> list[str]:
    """Cheap directory listing - does not parse any `.mat` file."""
    return DatasetLoader().list_trial_filenames(subject_id)


def load_single_trial(subject_id: str, filename: str) -> Trial | None:
    """
    Load exactly one `.mat` file on demand. Safe to call from an
    interactive UI - unlike `DatasetManager.load()`, this never touches
    the other subjects.
    """
    return DatasetLoader().load_trial(subject_id, filename)


# ----------------------------------------------------------------------
# Running the pipeline from the dashboard
# ----------------------------------------------------------------------


def run_full_pipeline(status_callback: Callable[[str], None] | None = None) -> PipelineResult:
    """
    Runs the same load -> validate -> EDA pipeline as `main.py` (via
    `src.pipeline.run_pipeline`) and returns the result, including the
    new experiment's `ExperimentManager`. This can take a while for the
    full 27-subject dataset - callers should show a spinner/status and
    call this from behind a button, not on every rerun.
    """

    return run_pipeline(status_callback=status_callback)
