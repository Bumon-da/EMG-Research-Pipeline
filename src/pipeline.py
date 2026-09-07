"""
Pipeline Orchestration

Single source of truth for "run the full pipeline" (load -> validate ->
EDA -> split -> preprocess -> extract features). Used by both `main.py`
(CLI) and the dashboard's "Run Pipeline Now" button, so the two entry
points can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from config.settings import EXCLUDE_REST
from src.data.datamodels import Subject
from src.data.splitter import SplitResult, repetition_split
from src.data.validator import DatasetValidator
from src.eda.analyzer import EDAAnalyzer
from src.features.extractor import FeatureExtractor, FeatureResult
from src.managers.dataset_manager import DatasetManager
from src.managers.experiment_manager import ExperimentManager
from src.managers.results_manager import ResultsManager
from src.preprocessing.preprocessor import PreprocessingResult, SignalPreprocessor

StatusCallback = Callable[[str], None]


@dataclass
class PipelineResult:
    experiment: ExperimentManager
    subjects: list[Subject]
    validation_df: pd.DataFrame
    dataset_summary_df: pd.DataFrame
    split_result: SplitResult
    preprocessing: PreprocessingResult
    features: FeatureResult


def run_pipeline(status_callback: StatusCallback | None = None) -> PipelineResult:
    """
    Run the full load -> validate -> EDA -> split -> preprocess pipeline
    into a new, isolated experiment folder.

    Args:
        status_callback: optional callable invoked with short progress
            strings (e.g. `logger.info` for the CLI, or a Streamlit
            `st.status` updater for the dashboard).
    """

    def report(message: str) -> None:
        if status_callback:
            status_callback(message)

    report("Creating experiment run...")
    experiment = ExperimentManager()
    results = ResultsManager(output_root=experiment.path)

    report("Loading dataset...")
    dataset = DatasetManager()
    subjects = dataset.load()

    report(f"Loaded {dataset.subject_count} subjects / {dataset.trial_count} trials. Validating...")
    validator = DatasetValidator(results=results)
    validation_df = validator.validate(subjects)

    report("Running EDA and generating figures...")
    eda = EDAAnalyzer(results=results)
    dataset_summary_df = eda.analyze(subjects)

    report("Splitting train/test by repetition...")
    # EXCLUDE_REST=True is a correctness fix, not just a volume/imbalance
    # optimization: NinaPro DB1's rest periods never fall in
    # DEFAULT_TEST_REPETITIONS, so with exclude_rest=False the test split
    # would contain zero rest windows while train is dominated by them -
    # see config/settings.py's EXCLUDE_REST docstring.
    split_result = repetition_split(subjects, exclude_rest=EXCLUDE_REST)

    report("Running signal preprocessing (normalization stats, rectification, windowing tally)...")
    preprocessing_result = SignalPreprocessor(results=results).preprocess(subjects, split_result)

    report("Extracting features (Atzori et al. 2014 DB1 baseline)...")
    feature_result = FeatureExtractor(results=results).extract(
        subjects, split_result, precomputed_stats=preprocessing_result.subject_stats
    )

    report("Saving run manifest...")
    experiment.save_manifest(
        extra={
            "dataset": {
                "subjects": dataset.subject_count,
                "trials": dataset.trial_count,
                "flagged_trials": (
                    int(validation_df["flagged"].sum()) if len(validation_df) else 0
                ),
            },
            "preprocessing": {
                "split_strategy": split_result.strategy,
                "train_samples": split_result.train_size,
                "test_samples": split_result.test_size,
                **preprocessing_result.summary,
            },
            "features": feature_result.summary,
        }
    )

    report("Pipeline complete.")

    return PipelineResult(
        experiment=experiment,
        subjects=subjects,
        validation_df=validation_df,
        dataset_summary_df=dataset_summary_df,
        split_result=split_result,
        preprocessing=preprocessing_result,
        features=feature_result,
    )
