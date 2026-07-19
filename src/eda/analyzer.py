"""
EDA Analyzer

Generates:
- Dataset summary
- Descriptive statistics
"""

from pathlib import Path

import pandas as pd

from config.settings import EDA_DIR
from config.logging_config import logger

from src.eda.statistics import Statistics


class EDAAnalyzer:

    def __init__(self):

        self.output_dir = Path(EDA_DIR)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def analyze(self, subjects):

        logger.info("Starting Exploratory Data Analysis...")

        summary = []

        total_subjects = len(subjects)
        total_trials = 0
        total_samples = 0

        for subject in subjects:

            subject_folder = self.output_dir / subject.subject_id

            subject_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            for trial in subject.trials:

                total_trials += 1
                total_samples += trial.samples

                stats = Statistics.descriptive_statistics(
                    trial.emg
                )

                stats.to_csv(
                    subject_folder /
                    f"{trial.filename}_statistics.csv"
                )

                summary.append({
                    "Subject": subject.subject_id,
                    "Trial": trial.filename,
                    "Samples": trial.samples,
                    "Channels": trial.channels
                })

        summary_df = pd.DataFrame(summary)

        summary_df.to_csv(
            self.output_dir / "dataset_summary.csv",
            index=False
        )

        print("\n")
        print("=" * 60)
        print("EDA SUMMARY")
        print("=" * 60)

        print(f"Subjects : {total_subjects}")
        print(f"Trials   : {total_trials}")
        print(f"Samples  : {total_samples:,}")

        print("\nOutput Folder")
        print("-" * 60)
        print(self.output_dir)

        logger.info("EDA Complete.")