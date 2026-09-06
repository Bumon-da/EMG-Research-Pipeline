import numpy as np

from src.data.validator import DatasetValidator
from src.managers.results_manager import ResultsManager
from tests.conftest import make_subject, make_trial


def make_validator(tmp_path):
    return DatasetValidator(results=ResultsManager(output_root=tmp_path))


def test_flags_nan_values(tmp_path):
    emg = np.ones((10, 2))
    emg[0, 0] = np.nan

    trial = make_trial(n_samples=10, n_channels=2, emg=emg)
    subject = make_subject("S1", trials=[trial])

    details = make_validator(tmp_path).validate([subject])

    row = details.iloc[0]
    assert row["nan_values"] == 1
    assert row["flagged"] is True or bool(row["flagged"]) is True


def test_flags_negative_values(tmp_path):
    # NinaPro DB1's emg field is a rectified, non-negative sensor output;
    # any negative value is a data-integrity problem worth surfacing.
    emg = np.ones((5, 2))
    emg[2, 1] = -0.5

    trial = make_trial(n_samples=5, n_channels=2, emg=emg)
    subject = make_subject("S1", trials=[trial])

    details = make_validator(tmp_path).validate([subject])

    assert details.iloc[0]["negative_values"] == 1


def test_flags_constant_channel(tmp_path):
    rng = np.random.default_rng(1)
    emg = rng.random((20, 3))
    emg[:, 1] = 0.42  # constant channel

    trial = make_trial(n_samples=20, n_channels=3, emg=emg)
    subject = make_subject("S1", trials=[trial])

    details = make_validator(tmp_path).validate([subject])

    assert details.iloc[0]["constant_channels"] == 1


def test_detects_duplicate_trials(tmp_path):
    rng = np.random.default_rng(2)
    emg = rng.random((15, 2))

    trial_a = make_trial(filename="a.mat", n_samples=15, n_channels=2, emg=emg.copy())
    trial_b = make_trial(filename="b.mat", n_samples=15, n_channels=2, emg=emg.copy())

    subject_a = make_subject("S1", trials=[trial_a])
    subject_b = make_subject("S2", trials=[trial_b])

    details = make_validator(tmp_path).validate([subject_a, subject_b])

    # The first occurrence should not be marked a duplicate; the second should.
    assert details.iloc[0]["is_duplicate"] == False
    assert details.iloc[1]["is_duplicate"] == True


def test_clean_trial_is_not_flagged(tmp_path):
    rng = np.random.default_rng(3)
    emg = rng.random((50, 4)) + 0.1  # avoid near-zero RMS on any channel

    trial = make_trial(n_samples=50, n_channels=4, emg=emg)
    subject = make_subject("S1", trials=[trial])

    details = make_validator(tmp_path).validate([subject])

    assert bool(details.iloc[0]["flagged"]) is False
