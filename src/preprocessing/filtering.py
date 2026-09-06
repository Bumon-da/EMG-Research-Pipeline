"""
Raw-sEMG Filtering

A generic bandpass + notch filter for raw, AC-coupled surface EMG. NOT
applied anywhere in this project's current NinaPro DB1 processing path -
DB1's `emg` field is already a rectified, non-negative sensor envelope
(see src/data/validator.py's module docstring and PROJECT_STATUS.md's Key
Finding #3), and applying a raw-sEMG filter to an already-rectified,
non-negative signal risks introducing artifacts (e.g. ringing near the
noise floor) rather than removing them.

This function exists for future raw-sEMG sources - e.g. real EMG hardware
feeding the closed-loop EMS system sketched in
`pipeline diagrams/ems_pipeline.xml` - where a genuine bandpass/notch
filter will be needed. It validates its own arguments against the
sampling rate's Nyquist limit: DB1's own configured settings
(SAMPLING_RATE=100, HIGHCUT=450, NOTCH_FREQ=50) are mathematically
invalid for a 100Hz signal (Nyquist=50Hz) and raise here - an additional,
concrete reason, beyond the signal-type mismatch, that this filter is
never invoked in the DB1 path.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from config.settings import HIGHCUT, LOWCUT, NOTCH_FREQ, SAMPLING_RATE


def bandpass_notch_filter(
    emg: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    lowcut: float = LOWCUT,
    highcut: float = HIGHCUT,
    notch_freq: float = NOTCH_FREQ,
    filter_order: int = 4,
    notch_quality: float = 30.0,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass followed by an IIR notch, applied per
    channel (axis=0). Intended for a whole trial at a genuine raw-sEMG
    sampling rate, not a ~20-sample post-windowing segment - `filtfilt`
    requires the input be longer than roughly 3x the combined filter
    length.

    Raises:
        ValueError: if `lowcut`/`highcut`/`notch_freq` are not all
            strictly between 0 and the Nyquist frequency
            (`sampling_rate / 2`), or `lowcut >= highcut`.
    """

    nyquist = sampling_rate / 2

    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"bandpass_notch_filter: lowcut={lowcut}, highcut={highcut} are not "
            f"valid for sampling_rate={sampling_rate} (Nyquist={nyquist}Hz) - "
            f"need 0 < lowcut < highcut < Nyquist."
        )

    if not (0 < notch_freq < nyquist):
        raise ValueError(
            f"bandpass_notch_filter: notch_freq={notch_freq} is not valid for "
            f"sampling_rate={sampling_rate} (Nyquist={nyquist}Hz) - need "
            f"0 < notch_freq < Nyquist."
        )

    b_band, a_band = butter(filter_order, [lowcut / nyquist, highcut / nyquist], btype="band")
    b_notch, a_notch = iirnotch(notch_freq / nyquist, notch_quality)

    filtered = filtfilt(b_band, a_band, emg, axis=0)
    filtered = filtfilt(b_notch, a_notch, filtered, axis=0)

    return filtered
