"""
Time-Domain Features

Stateless, vectorized feature functions operating on a
`(n_windows, channels, window_len)` array of windowed EMG - mirrors how
`src/eda/statistics.py`'s `Statistics` class is written (no I/O, no
`ResultsManager` dependency). This is the Atzori et al. 2014 NinaPro DB1
baseline feature set, minus the frequency-domain features (see
`src/features/report.py` for why those are dropped for this dataset).

MCR (not ZC): DB1's `emg` is a non-negative sensor envelope (see
`src/data/validator.py`), so literature zero-crossing-rate is identically
0 on the raw signal, and even after per-subject z-scoring, a single
200ms window frequently never crosses the SUBJECT's global mean - 18% of
windows were measured all-zero across every channel. `mcr()` instead
counts crossings of each WINDOW's own mean (a mean-crossing rate), which
eliminates the all-zero windows entirely. This is not literature ZC and
should not be compared to it - see `src/features/report.py`.
"""

from __future__ import annotations

import numpy as np


class TimeDomainFeatures:

    @staticmethod
    def rms(windows: np.ndarray) -> np.ndarray:
        """Root mean square, per (window, channel). Shape (n, ch, L) -> (n, ch)."""
        return np.sqrt(np.mean(np.square(windows), axis=-1))

    @staticmethod
    def mav(windows: np.ndarray) -> np.ndarray:
        """Mean absolute value, per (window, channel)."""
        return np.mean(np.abs(windows), axis=-1)

    @staticmethod
    def iemg(windows: np.ndarray) -> np.ndarray:
        """
        Integrated EMG (sum of absolute value), per (window, channel).
        On a fixed-length window this is exactly window_len * mav() - kept
        for literature fidelity, but perfectly collinear with MAV; drop
        one before fitting any covariance-based model (see
        src/features/report.py).
        """
        return np.sum(np.abs(windows), axis=-1)

    @staticmethod
    def wl(windows: np.ndarray) -> np.ndarray:
        """Waveform length: sum of absolute first differences."""
        return np.sum(np.abs(np.diff(windows, axis=-1)), axis=-1)

    @staticmethod
    def mcr(windows: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Mean-crossing rate: number of times consecutive samples, after
        subtracting the WINDOW's own mean, change sign. `threshold` is an
        optional deadzone (a crossing only counts if both demeaned samples
        exceed `threshold` in magnitude) - 0.0 (no deadzone) is the
        default since this signal is already per-subject normalized.

        Uses the product-sign test `(a * b) < 0` rather than comparing
        `np.sign(a) != np.sign(b)` - the latter manufactures spurious
        crossings whenever a sample lands exactly on the demeaned zero
        point (verified on real data: 37 spurious crossings out of 0 true
        ones on the raw, un-demeaned signal, from exact-zero quantization
        floor samples).
        """
        demeaned = windows - windows.mean(axis=-1, keepdims=True)
        a, b = demeaned[..., :-1], demeaned[..., 1:]

        crosses = (a * b) < 0
        if threshold > 0.0:
            crosses &= (np.abs(a) > threshold) & (np.abs(b) > threshold)

        return np.sum(crosses, axis=-1)

    @staticmethod
    def ssc(windows: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Slope sign changes: number of local extrema in the signal, i.e.
        sign changes in the first difference. Unaffected by whether the
        input is raw or normalized (confirmed bit-exact identical either
        way on real data) - z-scoring is an affine transform with a
        positive scale, which preserves the sign of every difference.
        """
        diff = np.diff(windows, axis=-1)
        a, b = diff[..., :-1], diff[..., 1:]

        changes = (a * b) < 0
        if threshold > 0.0:
            changes &= (np.abs(a) > threshold) & (np.abs(b) > threshold)

        return np.sum(changes, axis=-1)

    @staticmethod
    def hist(windows: np.ndarray, bin_edges: list[float]) -> np.ndarray:
        """
        Amplitude histogram per (window, channel), with fixed, shared bin
        edges (never per-window edges - that would make the feature
        incomparable across rows). `bin_edges` gives the interior
        boundaries only; the outer bins are unbounded (-inf/+inf), so
        `len(bin_edges) + 1` bins are produced.

        Returns shape (n_windows, channels, len(bin_edges) + 1), int
        counts (windows are small - 20 samples - so int8 covers the
        range).
        """
        edges = [-np.inf, *bin_edges, np.inf]
        n_bins = len(edges) - 1

        counts = np.empty((*windows.shape[:-1], n_bins), dtype=np.int8)
        for i in range(n_bins):
            lower, upper = edges[i], edges[i + 1]
            in_bin = (windows > lower) & (windows <= upper)
            counts[..., i] = np.sum(in_bin, axis=-1)

        return counts
