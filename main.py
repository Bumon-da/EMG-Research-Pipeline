"""
EMG Research Pipeline
Main Entry Point
"""

from __future__ import annotations

from config.logging_config import logger
from src.data.datamodels import Subject
from src.pipeline import run_pipeline


def print_dataset_overview(subjects: list[Subject]) -> None:
    """
    Print a quick overview of the loaded dataset.
    """

    if not subjects:
        print("No subjects loaded.")
        return

    first_subject = subjects[0]
    first_trial = first_subject.trials[0]

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Subjects Loaded : {len(subjects)}")
    print(f"First Subject   : {first_subject.subject_id}")
    print(f"Trials          : {first_subject.num_trials}")

    print("\nFirst Trial")
    print("-" * 60)
    print(f"Filename           : {first_trial.filename}")
    print(f"Samples            : {first_trial.samples}")
    print(f"Channels           : {first_trial.channels}")
    print(f"Has refined labels : {first_trial.has_refined_labels}")
    print(f"Has glove data     : {first_trial.has_glove}")


def main() -> None:
    """
    Execute the EMG Research Pipeline.
    """

    logger.info("=" * 60)
    logger.info("Starting EMG Research Pipeline")
    logger.info("=" * 60)

    result = run_pipeline(status_callback=logger.info)

    print_dataset_overview(result.subjects)

    logger.info("=" * 60)
    logger.info("Pipeline Completed Successfully")
    logger.info(f"All outputs for this run are under: {result.experiment.path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
