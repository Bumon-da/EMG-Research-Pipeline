"""
EMG Research Pipeline
Main Entry Point
"""

from config.logging_config import logger

from src.managers.dataset_manager import DatasetManager
from src.eda.validator import DatasetValidator
from src.eda.analyzer import EDAAnalyzer


def print_dataset_overview(subjects):
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
    print(f"Filename : {first_trial.filename}")
    print(f"Samples  : {first_trial.samples}")
    print(f"Channels : {first_trial.channels}")


def main():
    """
    Execute the EMG Research Pipeline.
    """

    logger.info("=" * 60)
    logger.info("Starting EMG Research Pipeline")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # Load Dataset
    # ----------------------------------------------------------

    dataset = DatasetManager()
    subjects = dataset.load()

    logger.info(f"Subjects Loaded : {dataset.subject_count}")
    logger.info(f"Trials Loaded   : {dataset.trial_count}")

    print_dataset_overview(subjects)

    # ----------------------------------------------------------
    # Dataset Validation
    # ----------------------------------------------------------

    validator = DatasetValidator()
    validator.validate(subjects)

    # ----------------------------------------------------------
    # Exploratory Data Analysis
    # ----------------------------------------------------------

    eda = EDAAnalyzer()
    eda.analyze(subjects)

    logger.info("=" * 60)
    logger.info("Pipeline Completed Successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()