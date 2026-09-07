"""
Signal Preprocessing Orchestrator

Turns the previously-unconsumed src/data/splitter.py into a real pipeline
stage: for every subject, cleans each trial's signal (Normalizer.clean),
derives per-subject normalization statistics from train-split samples
only, and segments every trial into label-uniform, split-pure windows
(src.preprocessing.segmentation). No bandpass/notch filter is applied -
see src/preprocessing/filtering.py's module docstring for why.

Only small, inspectable artifacts are persisted (per-subject/channel
stats, per-trial rectification counts, per-(subject,trial,exercise,label,
split) window tallies) - not a materialized copy of the ~12.5M-sample
dataset and not the full ~1.2M-window index. There is no consumer for
either yet; that is feature extraction's job (a later milestone), which
can slice `trial.emg[window.start:window.end]` and re-derive/apply
normalization stats on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config.logging_config import logger
from src.data.datamodels import Subject
from src.data.splitter import SplitResult
from src.managers.results_manager import ResultsManager
from src.preprocessing.normalization import Normalizer, SubjectNormalizationStats
from src.preprocessing.report import build_preprocessing_report
from src.preprocessing.segmentation import Window, segment_trial_detailed


@dataclass
class PreprocessingResult:
    normalization_stats_df: pd.DataFrame  # one row per (Subject, Channel)
    rectification_df: pd.DataFrame  # one row per (Subject, Trial)
    window_tally_df: pd.DataFrame  # one row per (Subject, Trial, Exercise, Label, Split)
    window_drop_summary_df: pd.DataFrame  # one row per (Subject, Trial)
    subject_stats: dict[str, SubjectNormalizationStats]  # keyed by subject_id - feature
    # extraction (src/features/extractor.py) consumes this so it can reuse this run's
    # stats instead of recomputing an identical scan over the same split_result.
    summary: dict[str, Any]  # native int/bool/str only - keeps JSON output faithful


class SignalPreprocessor:

    def __init__(self, results: ResultsManager | None = None) -> None:
        self.results = results or ResultsManager()

    def preprocess(self, subjects: list[Subject], split_result: SplitResult) -> PreprocessingResult:
        logger.info("Starting Signal Preprocessing...")

        stats_rows: list[dict] = []
        rectification_rows: list[dict] = []
        tally_rows: list[dict] = []
        drop_summary_rows: list[dict] = []
        subject_stats_by_id: dict[str, SubjectNormalizationStats] = {}

        subjects_with_zero_trials = 0
        subjects_with_train_fallback = 0
        total_negative_clipped = 0
        total_non_finite = 0

        for subject in subjects:
            if not subject.trials:
                subjects_with_zero_trials += 1
                logger.warning(f"{subject.subject_id}: no trials - skipped by preprocessing.")
                continue

            trial_records = []

            for trial in subject.trials:
                trial_split = split_result.get(subject.subject_id, trial.filename)
                if trial_split is None:
                    logger.warning(
                        f"{subject.subject_id}/{trial.filename}: no split entry - "
                        f"skipped by preprocessing."
                    )
                    continue

                clean_emg, negative_count, non_finite_count = Normalizer.clean(trial.emg)
                total_negative_clipped += negative_count
                total_non_finite += non_finite_count

                rectification_rows.append(
                    {
                        "Subject": subject.subject_id,
                        "Trial": trial.filename,
                        "NegativeSamplesClipped": negative_count,
                        "NonFiniteSamplesFound": non_finite_count,
                    }
                )

                trial_records.append((trial, clean_emg, trial_split))

            if not trial_records:
                continue

            subject_stats = Normalizer.compute_subject_stats(
                subject.subject_id,
                [(clean_emg, trial_split.train_mask) for _, clean_emg, trial_split in trial_records],
            )
            if subject_stats.used_test_fallback:
                subjects_with_train_fallback += 1

            subject_stats_by_id[subject.subject_id] = subject_stats

            for channel in range(subject_stats.channel_mean.shape[0]):
                stats_rows.append(
                    {
                        "Subject": subject.subject_id,
                        "Channel": channel + 1,
                        "Mean": round(float(subject_stats.channel_mean[channel]), 6),
                        "Std": round(float(subject_stats.channel_std[channel]), 6),
                        "TrainSamples": subject_stats.train_sample_count,
                        "UsedTestFallback": subject_stats.used_test_fallback,
                        "ZeroStdFlagged": channel in subject_stats.zero_std_channels,
                    }
                )

            for trial, _clean_emg, trial_split in trial_records:
                windows, counts = segment_trial_detailed(trial, trial_split)

                drop_summary_rows.append(
                    {
                        "Subject": subject.subject_id,
                        "Trial": trial.filename,
                        "Exercise": trial.exercise_id,
                        "TotalCandidates": counts.total_candidates,
                        "KeptTrain": counts.kept_train,
                        "KeptTest": counts.kept_test,
                        "DroppedTransition": counts.dropped_transition,
                        "DroppedBoundary": counts.dropped_boundary,
                    }
                )

                tally_rows.extend(self._tally_windows(subject.subject_id, trial, windows))

        normalization_stats_df = pd.DataFrame(stats_rows)
        rectification_df = pd.DataFrame(rectification_rows)
        window_tally_df = pd.DataFrame(tally_rows)
        window_drop_summary_df = pd.DataFrame(drop_summary_rows)

        summary = self._summary(
            subjects_processed=len({row["Subject"] for row in stats_rows}),
            subjects_with_zero_trials=subjects_with_zero_trials,
            subjects_with_train_fallback=subjects_with_train_fallback,
            total_negative_clipped=total_negative_clipped,
            total_non_finite=total_non_finite,
            window_drop_summary_df=window_drop_summary_df,
        )

        self.results.save_dataframe(
            dataframe=normalization_stats_df, folder="preprocessing", filename="normalization_stats.csv"
        )
        self.results.save_dataframe(
            dataframe=rectification_df, folder="preprocessing", filename="rectification_report.csv"
        )
        self.results.save_dataframe(
            dataframe=window_tally_df, folder="preprocessing", filename="window_tally.csv"
        )
        self.results.save_dataframe(
            dataframe=window_drop_summary_df, folder="preprocessing", filename="window_drop_summary.csv"
        )
        self.results.save_json(data=summary, folder="preprocessing", filename="preprocessing_summary.json")

        report_text = build_preprocessing_report(
            normalization_stats_df, rectification_df, window_tally_df, window_drop_summary_df, summary
        )
        self.results.save_text(text=report_text, folder="preprocessing", filename="preprocessing_report.txt")

        print("\n")
        print("=" * 60)
        print("SIGNAL PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Subjects processed             : {summary['subjects_processed']}")
        print(f"Subjects skipped (no trials)    : {summary['subjects_with_zero_trials']}")
        print(f"Subjects using test fallback    : {summary['subjects_with_train_fallback']}")
        print(f"Negative samples clipped        : {summary['total_negative_samples_clipped']:,}")
        print(f"Non-finite samples found        : {summary['total_non_finite_samples_found']:,}")
        print(
            f"Windows kept (train / test)     : "
            f"{summary['total_windows_train']:,} / {summary['total_windows_test']:,}"
        )
        print(
            f"Windows dropped (transition / boundary): "
            f"{summary['total_windows_dropped_transition']:,} / {summary['total_windows_dropped_boundary']:,}"
        )

        logger.info("Signal Preprocessing Complete.")

        return PreprocessingResult(
            normalization_stats_df=normalization_stats_df,
            rectification_df=rectification_df,
            window_tally_df=window_tally_df,
            window_drop_summary_df=window_drop_summary_df,
            subject_stats=subject_stats_by_id,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tally_windows(subject_id: str, trial, windows: list[Window]) -> list[dict]:
        if not windows:
            return []

        counts: dict[tuple[int, str], int] = {}
        for window in windows:
            key = (window.label, window.split)
            counts[key] = counts.get(key, 0) + 1

        exercise_id = trial.exercise_id

        return [
            {
                "Subject": subject_id,
                "Trial": trial.filename,
                "Exercise": exercise_id,
                "Label": label,
                "Split": split,
                "WindowCount": count,
            }
            for (label, split), count in sorted(counts.items())
        ]

    @staticmethod
    def _summary(
        subjects_processed: int,
        subjects_with_zero_trials: int,
        subjects_with_train_fallback: int,
        total_negative_clipped: int,
        total_non_finite: int,
        window_drop_summary_df: pd.DataFrame,
    ) -> dict[str, Any]:

        if window_drop_summary_df.empty:
            train_windows = test_windows = dropped_transition = dropped_boundary = 0
        else:
            train_windows = int(window_drop_summary_df["KeptTrain"].sum())
            test_windows = int(window_drop_summary_df["KeptTest"].sum())
            dropped_transition = int(window_drop_summary_df["DroppedTransition"].sum())
            dropped_boundary = int(window_drop_summary_df["DroppedBoundary"].sum())

        return {
            "subjects_processed": int(subjects_processed),
            "subjects_with_zero_trials": int(subjects_with_zero_trials),
            "subjects_with_train_fallback": int(subjects_with_train_fallback),
            "total_negative_samples_clipped": int(total_negative_clipped),
            "total_non_finite_samples_found": int(total_non_finite),
            "total_windows": train_windows + test_windows,
            "total_windows_train": train_windows,
            "total_windows_test": test_windows,
            "total_windows_dropped_transition": dropped_transition,
            "total_windows_dropped_boundary": dropped_boundary,
        }
