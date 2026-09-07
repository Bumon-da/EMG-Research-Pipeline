"""
Feature Extraction Orchestrator

Turns src/preprocessing/segmentation.py's segment_trial_detailed() and
src/preprocessing/normalization.py's Normalizer.apply() into real
consumers for the first time (both were dead code as of v0.5.0, built
for exactly this purpose). Per subject: clean -> per-subject/per-channel
z-score normalize (train-split only) -> segment into label-uniform,
split-pure windows -> extract the Atzori et al. 2014 DB1 baseline feature
set (minus frequency-domain features - see src/features/report.py) from
the NORMALIZED signal.

Normalization must happen before extraction: DB1's emg is non-negative,
so a crossing-rate feature computed on the raw signal is identically 0
for every window (see src/features/time_domain.py).

Normalization statistics are accepted as an optional `precomputed_stats`
argument (see `extract()`) rather than always recomputed: `src/pipeline.py`
passes the SAME run's `SignalPreprocessor.preprocess()` output, which was
already scanned against this exact `split_result` (including whatever
EXCLUDE_REST produced), so recomputing here would just repeat an identical
full-dataset scan. Recomputation only happens for a caller that omits
`precomputed_stats` (e.g. a test, or a future caller with a different
split) - never silently reuse stats derived from a DIFFERENT split_result,
since EXCLUDE_REST shifts the train-split mean/std used to compute them
(measured up to ~3x on a single channel).

Each trial's windows are extracted with one vectorized pass rather than
a per-window Python loop: np.lib.stride_tricks.sliding_window_view gives
a zero-copy view over every candidate window position, and fancy-
indexing by the kept windows' start offsets materializes only those
windows for that trial.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config.logging_config import logger
from config.settings import (
    EXERCISE_NUM_GESTURES,
    HIST_BIN_EDGES,
    MCR_THRESHOLD,
    SSC_THRESHOLD,
    WAVELET,
    WAVELET_LEVEL,
)
from src.data.datamodels import Subject, Trial
from src.data.splitter import SplitResult, TrialSplit
from src.features.report import build_features_report
from src.features.time_domain import TimeDomainFeatures
from src.features.wavelet import WaveletFeatures
from src.managers.results_manager import ResultsManager
from src.preprocessing.normalization import Normalizer, SubjectNormalizationStats
from src.preprocessing.segmentation import Window, segment_trial_detailed, window_samples


def exercise_label_offsets(exercise_num_gestures: dict[int, int] = EXERCISE_NUM_GESTURES) -> dict[int, int]:
    """
    Cumulative offset per exercise so a GlobalLabel = Label + offset
    column is unique across every exercise in the dataset (NinaPro DB1
    restarts gesture numbering at 1 for every exercise file - see
    config/settings.py's EXERCISE_NUM_GESTURES docstring). Exercises are
    processed in ascending order regardless of dict insertion order.
    """
    offsets: dict[int, int] = {}
    running = 0
    for exercise_id in sorted(exercise_num_gestures):
        offsets[exercise_id] = running
        running += exercise_num_gestures[exercise_id]
    return offsets


@dataclass
class FeatureResult:
    feature_manifest_df: pd.DataFrame  # one row per feature column: name, channel, group, dtype
    normalization_stats_df: pd.DataFrame  # one row per (Subject, Channel) - recomputed for this split
    summary: dict[str, Any]  # native int/bool/str only - keeps JSON output faithful


class FeatureExtractor:

    def __init__(self, results: ResultsManager | None = None) -> None:
        self.results = results or ResultsManager()
        self.window_len = window_samples()
        self.label_offsets = exercise_label_offsets()

    def extract(
        self,
        subjects: list[Subject],
        split_result: SplitResult,
        precomputed_stats: dict[str, SubjectNormalizationStats] | None = None,
    ) -> FeatureResult:
        """
        Args:
            precomputed_stats: per-subject normalization stats already
                computed against this SAME split_result (typically
                `SignalPreprocessor.preprocess(...).subject_stats` from the
                same pipeline run - see module docstring). A subject
                missing from this dict falls back to computing its own
                stats here, so partial dicts (e.g. in a test) are safe.
        """
        logger.info("Starting Feature Extraction...")
        start_time = time.monotonic()
        precomputed_stats = precomputed_stats or {}

        stats_rows: list[dict] = []
        subjects_processed = 0
        subjects_with_zero_trials = 0
        subjects_with_train_fallback = 0
        subjects_with_no_windows = 0
        total_windows = 0
        total_windows_train = 0
        total_windows_test = 0
        channel_counts: set[int] = set()

        for subject in subjects:
            if not subject.trials:
                subjects_with_zero_trials += 1
                logger.warning(f"{subject.subject_id}: no trials - skipped by feature extraction.")
                continue

            trial_records: list[tuple[Trial, np.ndarray, TrialSplit]] = []
            for trial in subject.trials:
                trial_split = split_result.get(subject.subject_id, trial.filename)
                if trial_split is None:
                    logger.warning(
                        f"{subject.subject_id}/{trial.filename}: no split entry - "
                        f"skipped by feature extraction."
                    )
                    continue

                clean_emg, _negative_count, _non_finite_count = Normalizer.clean(trial.emg)
                trial_records.append((trial, clean_emg, trial_split))

            if not trial_records:
                continue

            subject_stats = precomputed_stats.get(subject.subject_id) or Normalizer.compute_subject_stats(
                subject.subject_id,
                [(clean_emg, trial_split.train_mask) for _, clean_emg, trial_split in trial_records],
            )
            if subject_stats.used_test_fallback:
                subjects_with_train_fallback += 1

            channel_counts.add(subject_stats.channel_mean.shape[0])
            for channel in range(subject_stats.channel_mean.shape[0]):
                stats_rows.append(
                    {
                        "Subject": subject.subject_id,
                        "Channel": channel + 1,
                        "Mean": round(float(subject_stats.channel_mean[channel]), 6),
                        "Std": round(float(subject_stats.channel_std[channel]), 6),
                        "TrainSamples": subject_stats.train_sample_count,
                        "UsedTestFallback": subject_stats.used_test_fallback,
                    }
                )

            subject_frames: list[pd.DataFrame] = []
            for trial, clean_emg, trial_split in trial_records:
                windows, counts = segment_trial_detailed(trial, trial_split)
                if not windows:
                    continue

                trial_df = self._extract_trial(subject.subject_id, trial, clean_emg, subject_stats, windows)
                subject_frames.append(trial_df)

                total_windows += len(windows)
                total_windows_train += counts.kept_train
                total_windows_test += counts.kept_test

            if not subject_frames:
                subjects_with_no_windows += 1
                logger.warning(f"{subject.subject_id}: no usable windows - skipped by feature extraction.")
                continue

            subject_df = pd.concat(subject_frames, ignore_index=True)
            self.results.save_parquet(
                dataframe=subject_df,
                folder="features",
                filename=f"{subject.subject_id}_features.parquet",
            )
            subjects_processed += 1

        if len(channel_counts) > 1:
            logger.warning(
                f"Feature extraction saw inconsistent per-subject channel counts "
                f"{sorted(channel_counts)} - feature_manifest.csv reflects the max "
                f"({max(channel_counts)}); subjects with fewer channels will have "
                f"fewer columns in their own Parquet file than the manifest lists."
            )

        normalization_stats_df = pd.DataFrame(stats_rows)
        feature_manifest_df = self._build_feature_manifest(max(channel_counts, default=0))

        elapsed_seconds = round(time.monotonic() - start_time, 2)

        summary = {
            "subjects_processed": int(subjects_processed),
            "subjects_with_zero_trials": int(subjects_with_zero_trials),
            "subjects_with_train_fallback": int(subjects_with_train_fallback),
            "subjects_with_no_windows": int(subjects_with_no_windows),
            "total_windows": int(total_windows),
            "total_windows_train": int(total_windows_train),
            "total_windows_test": int(total_windows_test),
            "feature_columns": int(len(feature_manifest_df)),
            "wavelet": WAVELET,
            "wavelet_level": int(WAVELET_LEVEL),
            "hist_bins": int(len(HIST_BIN_EDGES) + 1),
            "elapsed_seconds": elapsed_seconds,
        }

        self.results.save_dataframe(
            dataframe=normalization_stats_df, folder="features", filename="normalization_stats.csv"
        )
        self.results.save_dataframe(
            dataframe=feature_manifest_df, folder="features", filename="feature_manifest.csv"
        )
        self.results.save_json(data=summary, folder="features", filename="feature_summary.json")

        report_text = self._build_report(summary, feature_manifest_df)
        self.results.save_text(text=report_text, folder="features", filename="features_report.txt")

        print("\n")
        print("=" * 60)
        print("FEATURE EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Subjects processed              : {summary['subjects_processed']}")
        print(f"Feature columns                  : {summary['feature_columns']}")
        print(
            f"Windows kept (train / test)      : "
            f"{summary['total_windows_train']:,} / {summary['total_windows_test']:,}"
        )
        print(f"Elapsed                          : {summary['elapsed_seconds']}s")

        logger.info("Feature Extraction Complete.")

        return FeatureResult(
            feature_manifest_df=feature_manifest_df,
            normalization_stats_df=normalization_stats_df,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Per-trial extraction
    # ------------------------------------------------------------------

    def _extract_trial(
        self,
        subject_id: str,
        trial: Trial,
        clean_emg: np.ndarray,
        subject_stats: SubjectNormalizationStats,
        windows: list[Window],
    ) -> pd.DataFrame:
        normalized = Normalizer.apply(clean_emg, subject_stats).astype(np.float32)

        views = np.lib.stride_tricks.sliding_window_view(normalized, self.window_len, axis=0)
        starts = np.fromiter((w.start for w in windows), dtype=np.intp, count=len(windows))
        selected = views[starts]  # (n_kept, channels, window_len) float32, a copy

        n_channels = selected.shape[1]

        metadata_df = self._build_metadata(subject_id, trial, windows, starts)
        feature_df = self._build_features(selected, n_channels)

        return pd.concat([metadata_df, feature_df], axis=1)

    def _build_metadata(
        self, subject_id: str, trial: Trial, windows: list[Window], starts: np.ndarray
    ) -> pd.DataFrame:
        exercise_id = trial.exercise_id
        offset = self.label_offsets.get(exercise_id)

        labels = np.fromiter((w.label for w in windows), dtype=np.int16, count=len(windows))
        splits = [w.split for w in windows]

        if offset is None:
            # Unknown/unparseable exercise_id (see Trial.exercise_id) - -1 is a
            # sentinel that can never equal a real GlobalLabel (offsets + labels
            # are always >= 0), so this row is visibly unusable rather than
            # silently colliding with exercise 1's real GlobalLabel range.
            logger.warning(
                f"{subject_id}/{trial.filename}: unrecognized exercise_id={exercise_id!r} - "
                f"GlobalLabel set to -1 sentinel for this trial's windows."
            )
            global_labels = np.full(len(labels), -1, dtype=np.int8)
        else:
            global_labels = (labels + offset).astype(np.int8)

        return pd.DataFrame(
            {
                "Subject": subject_id,
                "Trial": trial.filename,
                "Exercise": np.int8(exercise_id) if exercise_id is not None else np.int8(-1),
                "Label": labels.astype(np.int8),
                "GlobalLabel": global_labels,
                "Split": pd.Categorical(splits, categories=["train", "test"]),
                "Start": starts.astype(np.int32),
            }
        )

    def _build_features(self, selected: np.ndarray, n_channels: int) -> pd.DataFrame:
        columns: dict[str, np.ndarray] = {}

        time_domain = {
            "RMS": TimeDomainFeatures.rms(selected).astype(np.float32),
            "MAV": TimeDomainFeatures.mav(selected).astype(np.float32),
            "WL": TimeDomainFeatures.wl(selected).astype(np.float32),
            "IEMG": TimeDomainFeatures.iemg(selected).astype(np.float32),
            "MCR": TimeDomainFeatures.mcr(selected, threshold=MCR_THRESHOLD).astype(np.int8),
            "SSC": TimeDomainFeatures.ssc(selected, threshold=SSC_THRESHOLD).astype(np.int8),
        }
        for name, values in time_domain.items():
            for channel in range(n_channels):
                columns[f"Ch{channel + 1}_{name}"] = values[:, channel]

        hist_counts = TimeDomainFeatures.hist(selected, HIST_BIN_EDGES)  # (n, ch, n_bins)
        n_bins = hist_counts.shape[-1]
        for channel in range(n_channels):
            for b in range(n_bins):
                columns[f"Ch{channel + 1}_HIST_b{b}"] = hist_counts[:, channel, b]

        dwt_bands = WaveletFeatures.mdwt(selected, WAVELET, WAVELET_LEVEL)  # (n, ch, n_bands)
        band_names = WaveletFeatures.band_names(WAVELET_LEVEL)
        for channel in range(n_channels):
            for b, band_name in enumerate(band_names):
                columns[f"Ch{channel + 1}_DWT_{band_name}"] = dwt_bands[:, channel, b].astype(np.float32)

        return pd.DataFrame(columns)

    # ------------------------------------------------------------------
    # Manifest / report
    # ------------------------------------------------------------------

    def _build_feature_manifest(self, n_channels: int) -> pd.DataFrame:
        if n_channels == 0:
            return pd.DataFrame(columns=["Column", "Channel", "Group", "Dtype"])

        rows: list[dict] = []

        for name in ["RMS", "MAV", "WL", "IEMG"]:
            for channel in range(n_channels):
                rows.append(
                    {"Column": f"Ch{channel + 1}_{name}", "Channel": channel + 1, "Group": "time_domain", "Dtype": "float32"}
                )

        for name in ["MCR", "SSC"]:
            for channel in range(n_channels):
                rows.append(
                    {"Column": f"Ch{channel + 1}_{name}", "Channel": channel + 1, "Group": "time_domain", "Dtype": "int8"}
                )

        n_bins = len(HIST_BIN_EDGES) + 1
        for channel in range(n_channels):
            for b in range(n_bins):
                rows.append(
                    {"Column": f"Ch{channel + 1}_HIST_b{b}", "Channel": channel + 1, "Group": "histogram", "Dtype": "int8"}
                )

        for channel in range(n_channels):
            for band_name in WaveletFeatures.band_names(WAVELET_LEVEL):
                rows.append(
                    {
                        "Column": f"Ch{channel + 1}_DWT_{band_name}",
                        "Channel": channel + 1,
                        "Group": "wavelet",
                        "Dtype": "float32",
                    }
                )

        return pd.DataFrame(rows)

    def _build_report(self, summary: dict[str, Any], feature_manifest_df: pd.DataFrame) -> str:
        return build_features_report(summary, feature_manifest_df)
