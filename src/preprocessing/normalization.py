"""
Signal Normalization

Per-subject, per-channel z-score normalization, computed only from each
subject's TRAIN-split samples (a `TrialSplit.train_mask` from
src.data.splitter) so that test-split values never influence the
statistics used to scale them. The same statistics are then applied to
both that subject's train and test rows - this makes src/data/splitter.py
(implemented since v0.4.0 but never consumed until now) a required input
to normalization, not an optional one.

Also provides a defensive rectification guard (`Normalizer.clean`).
NinaPro DB1's `emg` field is already a rectified, non-negative sensor
envelope (see src/data/validator.py's module docstring), so this is not a
real rectifier - it only clips stray negative samples (an integrity issue
per DatasetValidator) and neutralizes non-finite samples, so a single
corrupted trial can't silently poison a whole subject's statistics
(np.sum/np.mean propagate NaN across an entire reduction, not just the
affected cell).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config.logging_config import logger


@dataclass
class SubjectNormalizationStats:
    """
    Per-channel normalization parameters for one subject, derived from
    that subject's train-split samples only.
    """

    subject_id: str
    channel_mean: np.ndarray  # (channels,)
    channel_std: np.ndarray  # (channels,); near-zero std floored to 1.0
    train_sample_count: int
    zero_std_channels: list[int] = field(default_factory=list)
    used_test_fallback: bool = False


class Normalizer:
    """
    Stateless per-channel signal cleaning/normalization. Every method
    operates on a single (samples, channels) array and returns a new
    array or plain values - no I/O, no ResultsManager dependency (mirrors
    src/eda/statistics.py's Statistics class).
    """

    # A channel whose train-split std is at or below this is treated as
    # constant for normalization purposes and floored to 1.0 rather than
    # divided by near-zero (sklearn's StandardScaler convention). Kept as
    # its own constant rather than importing
    # DatasetValidator.CONSTANT_CHANNEL_STD_EPS - the two checks serve
    # different purposes (integrity flag vs. divide-by-zero guard) and
    # shouldn't be coupled just because they currently share a value.
    ZERO_STD_EPS = 1e-9

    @staticmethod
    def clean(emg: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        Clip negative samples to 0 and zero out non-finite (NaN/Inf)
        samples. Returns (clean_array, negative_count, non_finite_count).
        Returns the original array object unchanged (no copy) when both
        counts are 0, which is the expected common case for this dataset.
        """

        non_finite_mask = ~np.isfinite(emg)
        negative_mask = emg < 0

        non_finite_count = int(np.sum(non_finite_mask))
        negative_count = int(np.sum(negative_mask & ~non_finite_mask))

        if non_finite_count == 0 and negative_count == 0:
            return emg, 0, 0

        clean = emg.copy()
        clean[non_finite_mask] = 0.0
        clean[negative_mask & ~non_finite_mask] = 0.0

        return clean, negative_count, non_finite_count

    @staticmethod
    def compute_subject_stats(
        subject_id: str,
        cleaned_trials: list[tuple[np.ndarray, np.ndarray]],
    ) -> SubjectNormalizationStats:
        """
        Per-channel mean/std from train_mask-selected rows only, streamed
        one trial at a time via a running sum/sum-of-squares accumulator
        (never concatenating a whole subject's rows into one array).

        Args:
            cleaned_trials: [(clean_emg, train_mask), ...], one entry per
                trial belonging to this subject. `clean_emg` should
                already have gone through `Normalizer.clean`.

        If every trial's train_mask is empty (e.g. this subject was
        entirely held out by `subject_split`), falls back to that
        subject's full data (train+test) instead of producing NaN stats -
        `used_test_fallback` records this so callers/reports can flag it.
        """

        if not cleaned_trials:
            raise ValueError(f"compute_subject_stats: no trials given for subject {subject_id!r}")

        channels = cleaned_trials[0][0].shape[1]

        total_train_samples = sum(int(mask.sum()) for _, mask in cleaned_trials)
        used_test_fallback = total_train_samples == 0

        n = 0
        total = np.zeros(channels)
        total_sq = np.zeros(channels)

        for clean_emg, train_mask in cleaned_trials:
            rows = clean_emg if used_test_fallback else clean_emg[train_mask]
            if rows.shape[0] == 0:
                continue
            n += rows.shape[0]
            total += rows.sum(axis=0)
            total_sq += np.square(rows).sum(axis=0)

        mean = total / n
        variance = np.maximum(total_sq / n - np.square(mean), 0.0)
        std = np.sqrt(variance)

        zero_std_channels = [int(i) for i in np.where(std <= Normalizer.ZERO_STD_EPS)[0]]
        if zero_std_channels:
            std = std.copy()
            std[zero_std_channels] = 1.0

        if used_test_fallback:
            logger.warning(
                f"{subject_id}: no train-split samples available - normalization "
                f"stats computed from this subject's full data instead."
            )

        return SubjectNormalizationStats(
            subject_id=subject_id,
            channel_mean=mean,
            channel_std=std,
            train_sample_count=n,
            zero_std_channels=zero_std_channels,
            used_test_fallback=used_test_fallback,
        )

    @staticmethod
    def apply(emg: np.ndarray, stats: SubjectNormalizationStats) -> np.ndarray:
        """
        Per-channel z-score: (emg - mean) / std, using stats derived from
        this subject's TRAIN split (see `compute_subject_stats`) - applied
        identically to train and test rows, so test rows are always
        normalized using train-derived statistics, never their own.
        """

        return (emg - stats.channel_mean) / stats.channel_std
