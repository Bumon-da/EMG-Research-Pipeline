import numpy as np
import pytest

from src.preprocessing.filtering import bandpass_notch_filter


def test_bandpass_notch_filter_rejects_db1_config_combination():
    # DB1's own real settings: SAMPLING_RATE=100 -> Nyquist=50Hz, which
    # HIGHCUT=450 exceeds and NOTCH_FREQ=50 sits exactly on - this is
    # exactly why the DB1 processing path never calls this function.
    emg = np.random.default_rng(0).random((1000, 2))

    with pytest.raises(ValueError):
        bandpass_notch_filter(emg, sampling_rate=100, lowcut=20, highcut=450, notch_freq=50)


def test_bandpass_notch_filter_rejects_notch_at_or_above_nyquist():
    emg = np.random.default_rng(0).random((1000, 2))

    with pytest.raises(ValueError):
        bandpass_notch_filter(emg, sampling_rate=1000, lowcut=20, highcut=450, notch_freq=500)


def test_bandpass_notch_filter_runs_on_valid_high_rate_signal():
    rng = np.random.default_rng(0)
    emg = rng.random((2000, 2))

    filtered = bandpass_notch_filter(emg, sampling_rate=1000, lowcut=20, highcut=450, notch_freq=50)

    assert filtered.shape == emg.shape


def test_bandpass_notch_filter_attenuates_frequency_outside_passband():
    sampling_rate = 1000
    t = np.arange(0, 2.0, 1 / sampling_rate)

    # 5 Hz sine, well below the 20 Hz lowcut - should be heavily attenuated.
    low_freq_signal = np.sin(2 * np.pi * 5 * t)
    emg = np.column_stack([low_freq_signal, low_freq_signal])

    filtered = bandpass_notch_filter(emg, sampling_rate=sampling_rate, lowcut=20, highcut=450, notch_freq=50)

    original_amplitude = np.abs(low_freq_signal).max()
    filtered_amplitude = np.abs(filtered[:, 0]).max()

    assert filtered_amplitude < original_amplitude * 0.5
