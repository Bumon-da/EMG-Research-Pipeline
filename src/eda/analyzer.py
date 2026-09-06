"""
EDA Analyzer

Generates:
- Dataset summary
- Per-trial descriptive statistics
- A sample of diagnostic figures
- A short text report
"""

from __future__ import annotations

import pandas as pd

from config.logging_config import logger
from src.data.datamodels import Subject
from src.eda.report import build_eda_report
from src.eda.statistics import Statistics
from src.eda.visualization import Visualizer
from src.managers.results_manager import ResultsManager


class EDAAnalyzer:

    # Generating full figures (signal/histogram/boxplot/correlation) for
    # every trial in the dataset produces far more images than anyone
    # will look at. Instead, detailed per-trial figures are generated for
    # only the first trial of the first N subjects; the two dataset-wide
    # figures (subject comparison, gesture distribution) still cover
    # every subject.
    DETAILED_PLOT_SUBJECT_LIMIT = 3

    def __init__(self, results: ResultsManager | None = None) -> None:
        self.results = results or ResultsManager()
        self.visualizer = Visualizer(self.results)

    def analyze(self, subjects: list[Subject]) -> pd.DataFrame:

        logger.info("Starting Exploratory Data Analysis...")

        summary = []
        sampled_trial_filenames: list[str] = []

        total_subjects = len(subjects)
        total_trials = 0
        total_samples = 0

        for subject_index, subject in enumerate(subjects):

            for trial_index, trial in enumerate(subject.trials):

                total_trials += 1
                total_samples += trial.samples

                stats = Statistics.descriptive_statistics(trial.emg)

                self.results.save_dataframe(
                    dataframe=stats,
                    folder=f"eda/{subject.subject_id}",
                    filename=f"{trial.filename}_statistics.csv",
                    index=True,
                )

                summary.append(
                    {
                        "Subject": subject.subject_id,
                        "Trial": trial.filename,
                        "Samples": trial.samples,
                        "Channels": trial.channels,
                        "HasRefinedLabels": trial.has_refined_labels,
                        "HasGlove": trial.has_glove,
                    }
                )

                is_first_trial = trial_index == 0
                if is_first_trial and subject_index < self.DETAILED_PLOT_SUBJECT_LIMIT:
                    self._plot_trial(trial)
                    sampled_trial_filenames.append(f"{subject.subject_id}/{trial.filename}")

        summary_df = pd.DataFrame(summary)

        self.results.save_dataframe(
            dataframe=summary_df,
            folder="eda",
            filename="dataset_summary.csv",
        )

        if subjects:
            self.visualizer.plot_subject_comparison(subjects)
            self.visualizer.plot_gesture_distribution(subjects)

        report_text = build_eda_report(
            total_subjects=total_subjects,
            total_trials=total_trials,
            total_samples=total_samples,
            sampled_trial_filenames=sampled_trial_filenames,
            summary_df=summary_df,
        )

        self.results.save_text(
            text=report_text,
            folder="eda",
            filename="eda_report.txt",
        )

        print("\n")
        print("=" * 60)
        print("EDA SUMMARY")
        print("=" * 60)

        print(f"Subjects : {total_subjects}")
        print(f"Trials   : {total_trials}")
        print(f"Samples  : {total_samples:,}")

        logger.info("EDA Complete.")

        return summary_df

    def _plot_trial(self, trial) -> None:
        try:
            self.visualizer.plot_raw_signal(trial)
            self.visualizer.plot_channel_histograms(trial)
            self.visualizer.plot_channel_boxplots(trial)
            self.visualizer.plot_channel_correlation_heatmap(trial)
        except Exception:
            logger.exception(f"Failed to generate figures for {trial.filename}")
