import numpy as np

from tests.conftest import make_trial


def test_labels_prefers_restimulus_over_stimulus():
    trial = make_trial(
        n_samples=5,
        stimulus=np.array([1, 1, 1, 1, 1]),
        restimulus=np.array([0, 0, 1, 1, 1]),
    )

    assert np.array_equal(trial.labels, trial.restimulus)
    assert not np.array_equal(trial.labels, trial.stimulus)


def test_labels_falls_back_to_stimulus_when_no_restimulus():
    trial = make_trial(n_samples=5, stimulus=np.array([2, 2, 2, 2, 2]))

    assert trial.restimulus is None
    assert np.array_equal(trial.labels, trial.stimulus)
    assert trial.has_refined_labels is False


def test_reps_prefers_rerepetition():
    trial = make_trial(
        n_samples=4,
        repetition=np.array([1, 1, 1, 1]),
        rerepetition=np.array([1, 1, 2, 2]),
    )

    assert np.array_equal(trial.reps, trial.rerepetition)


def test_gesture_counts():
    trial = make_trial(
        n_samples=6,
        stimulus=np.array([0, 0, 1, 1, 1, 2]),
    )

    counts = trial.gesture_counts

    assert counts == {0: 2, 1: 3, 2: 1}


def test_has_glove_reflects_presence():
    with_glove = make_trial(n_samples=3, glove=np.zeros((3, 22)))
    without_glove = make_trial(n_samples=3)

    assert with_glove.has_glove is True
    assert without_glove.has_glove is False
