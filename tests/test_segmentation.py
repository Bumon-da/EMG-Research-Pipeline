import numpy as np

from src.data.splitter import TrialSplit
from src.preprocessing.segmentation import (
    Window,
    segment_trial,
    segment_trial_detailed,
    step_samples,
    window_samples,
)
from tests.conftest import make_trial


def make_split(n_samples, train_mask=None, test_mask=None):
    if train_mask is None:
        train_mask = np.ones(n_samples, dtype=bool)
    if test_mask is None:
        test_mask = ~train_mask
    return TrialSplit(subject_id="S1", trial_filename="t.mat", train_mask=train_mask, test_mask=test_mask)


def test_window_samples_and_step_samples_match_config_defaults():
    assert window_samples() == 20
    assert step_samples() == 10


def test_segment_trial_produces_expected_window_count():
    n_samples = 100
    trial = make_trial(
        n_samples=n_samples,
        n_channels=2,
        restimulus=np.ones(n_samples, dtype=int),
        rerepetition=np.ones(n_samples, dtype=int),
    )
    trial_split = make_split(n_samples)

    windows, counts = segment_trial_detailed(
        trial, trial_split, window_size_ms=200, window_overlap=0.5, sampling_rate=100
    )

    expected = (n_samples - 20) // 10 + 1  # 9
    assert len(windows) == expected
    assert counts.total_candidates == expected
    assert counts.kept_train == expected
    assert counts.dropped_transition == 0
    assert counts.dropped_boundary == 0


def test_segment_trial_drops_windows_spanning_label_transition():
    n_samples = 40
    labels = np.array([1] * 20 + [2] * 20)
    trial = make_trial(
        n_samples=n_samples, n_channels=2, restimulus=labels, rerepetition=np.ones(n_samples, dtype=int)
    )
    trial_split = make_split(n_samples)

    windows, counts = segment_trial_detailed(
        trial, trial_split, window_size_ms=200, window_overlap=0.5, sampling_rate=100
    )

    # Candidates at starts 0, 10, 20. start=10 -> [10:30) spans the
    # label-1/label-2 transition at sample 20 -> dropped.
    assert counts.total_candidates == 3
    assert counts.dropped_transition == 1
    assert len(windows) == 2


def test_segment_trial_drops_windows_spanning_split_boundary_even_without_label_change():
    n_samples = 40
    labels = np.ones(n_samples, dtype=int)  # uniform label throughout

    trial = make_trial(
        n_samples=n_samples, n_channels=2, restimulus=labels, rerepetition=np.ones(n_samples, dtype=int)
    )
    # Split boundary at sample 20, independent of the (never-changing) label.
    train_mask = np.array([True] * 20 + [False] * 20)
    trial_split = make_split(n_samples, train_mask=train_mask, test_mask=~train_mask)

    windows, counts = segment_trial_detailed(
        trial, trial_split, window_size_ms=200, window_overlap=0.5, sampling_rate=100
    )

    # start=0 -> [0:20) all train -> kept. start=10 -> [10:30) straddles
    # the boundary at 20 -> dropped, even though the label never changes.
    # start=20 -> [20:40) all test -> kept.
    assert counts.total_candidates == 3
    assert counts.dropped_boundary == 1
    assert counts.dropped_transition == 0
    assert counts.kept_train == 1
    assert counts.kept_test == 1


def test_segment_trial_keeps_rest_labeled_windows():
    n_samples = 20
    labels = np.zeros(n_samples, dtype=int)
    trial = make_trial(
        n_samples=n_samples, n_channels=2, restimulus=labels, rerepetition=np.ones(n_samples, dtype=int)
    )
    trial_split = make_split(n_samples)

    windows = segment_trial(trial, trial_split, window_size_ms=200, window_overlap=0.5, sampling_rate=100)

    assert len(windows) == 1
    assert windows[0].label == 0


def test_segment_trial_returns_empty_list_for_trial_shorter_than_window():
    n_samples = 10
    trial = make_trial(
        n_samples=n_samples,
        n_channels=2,
        restimulus=np.ones(n_samples, dtype=int),
        rerepetition=np.ones(n_samples, dtype=int),
    )
    trial_split = make_split(n_samples)

    windows = segment_trial(trial, trial_split, window_size_ms=200, window_overlap=0.5, sampling_rate=100)

    assert windows == []


def test_window_carries_no_copied_emg_and_slicing_is_a_view():
    assert set(Window.__dataclass_fields__.keys()) == {"start", "end", "label", "split"}

    n_samples = 20
    trial = make_trial(
        n_samples=n_samples,
        n_channels=2,
        restimulus=np.ones(n_samples, dtype=int),
        rerepetition=np.ones(n_samples, dtype=int),
    )
    trial_split = make_split(n_samples)
    windows = segment_trial(trial, trial_split, window_size_ms=200, window_overlap=0.5, sampling_rate=100)

    sliced = trial.emg[windows[0].start : windows[0].end]
    assert np.shares_memory(trial.emg, sliced)


def test_segment_trial_reshapes_2d_uint8_labels():
    n_samples = 20
    labels_2d = np.ones((n_samples, 1), dtype=np.uint8)
    reps_2d = np.ones((n_samples, 1), dtype=np.uint8)
    trial = make_trial(n_samples=n_samples, n_channels=2, restimulus=labels_2d, rerepetition=reps_2d)
    trial_split = make_split(n_samples)

    windows = segment_trial(trial, trial_split, window_size_ms=200, window_overlap=0.5, sampling_rate=100)

    assert len(windows) == 1
    assert windows[0].label == 1


def test_segment_trial_detailed_counts_are_internally_consistent():
    n_samples = 40
    labels = np.array([1] * 20 + [2] * 20)
    trial = make_trial(
        n_samples=n_samples, n_channels=2, restimulus=labels, rerepetition=np.ones(n_samples, dtype=int)
    )
    trial_split = make_split(n_samples)

    _windows, counts = segment_trial_detailed(
        trial, trial_split, window_size_ms=200, window_overlap=0.5, sampling_rate=100
    )

    assert counts.total_candidates == (
        counts.kept_train + counts.kept_test + counts.dropped_transition + counts.dropped_boundary
    )
