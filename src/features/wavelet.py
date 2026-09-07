"""
Wavelet Features

Marginal discrete wavelet transform (mDWT): per-band energy (sum of
absolute coefficients) from a multi-level DWT decomposition, per
(window, channel).

Atzori et al. 2014 used db7 at 5 levels on much longer windows. Neither
is possible here: `pywt.dwt_max_level(20, db7)` is 0 - db7's 14-tap
filter cannot decompose a 20-sample window at all. `db2` (4-tap) is the
shortest common wavelet that still permits a useful decomposition,
giving `dwt_max_level(20, db2) == 2` (confirmed via pywt). This is a
deliberate deviation from the paper, not an oversight - see
`src/features/report.py`.
"""

from __future__ import annotations

import numpy as np
import pywt


class WaveletFeatures:

    @staticmethod
    def mdwt(windows: np.ndarray, wavelet: str, level: int) -> np.ndarray:
        """
        Per-band marginal energy from a `level`-level DWT, per
        (window, channel). Input shape (n_windows, channels, window_len);
        output shape (n_windows, channels, level + 1) - one column per
        detail band plus the final approximation band, ordered as
        `pywt.wavedec` returns them: [cA_level, cD_level, ..., cD_1].

        Vectorizes directly over the leading (window, channel) axes via
        `axis=-1` - no per-window Python loop. `mode="symmetric"` is pinned
        explicitly (PyWavelets' own default) rather than left implicit, so
        a future PyWavelets version changing its default boundary-padding
        mode can't silently shift every DWT feature - see test_wavelet.py.
        """
        coeffs = pywt.wavedec(windows, wavelet, level=level, axis=-1, mode="symmetric")

        bands = [np.sum(np.abs(band), axis=-1) for band in coeffs]

        return np.stack(bands, axis=-1)

    @staticmethod
    def band_names(level: int) -> list[str]:
        """
        Column-name suffixes matching `mdwt()`'s band order:
        [f"A{level}", f"D{level}", ..., "D1"].
        """
        return [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
