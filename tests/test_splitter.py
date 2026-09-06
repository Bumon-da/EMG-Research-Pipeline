import numpy as np

from src.data.splitter import repetition_split, subject_split
from tests.conftest import make_subject, make_trial


def test_repetition_split_partitions_without_overlap():
    trial = make_trial(
        n_samples=10,
        repetition=np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
        rerepetition=np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
        stimulus=np.array([1] * 10),
        restimulus=np.array([1] * 10),
    )
    subject = make_subject("S1", trials=[trial])

    result = repetition_split([subject], test_repetitions=[2, 5])
    trial_split = result.get("S1", trial.filename)

    # Every sample lands in exactly one of train/test.
    assert not np.any(trial_split.train_mask & trial_split.test_mask)
    assert np.all(trial_split.train_mask | trial_split.test_mask)

    assert trial_split.test_size == 4  # reps 2 and 5, two samples each
    assert trial_split.train_size == 6
    assert result.train_size == 6
    assert result.test_size == 4


def test_repetition_split_exclude_rest():
    trial = make_trial(
        n_samples=6,
        repetition=np.array([1, 1, 2, 2, 3, 3]),
        rerepetition=np.array([1, 1, 2, 2, 3, 3]),
        stimulus=np.array([0, 0, 1, 1, 2, 2]),
        restimulus=np.array([0, 0, 1, 1, 2, 2]),
    )
    subject = make_subject("S1", trials=[trial])

    result = repetition_split([subject], test_repetitions=[2], exclude_rest=True)
    trial_split = result.get("S1", trial.filename)

    # The two rest samples (label 0, repetition 1) should be excluded
    # from both masks entirely.
    assert trial_split.train_size + trial_split.test_size == 4


def test_subject_split_holds_out_whole_subjects():
    s1 = make_subject("S1", trials=[make_trial(filename="s1.mat", n_samples=5)])
    s2 = make_subject("S2", trials=[make_trial(filename="s2.mat", n_samples=5)])

    result = subject_split([s1, s2], test_subject_ids=["S2"])

    s1_split = result.get("S1", "s1.mat")
    s2_split = result.get("S2", "s2.mat")

    assert s1_split.train_size == 5
    assert s1_split.test_size == 0

    assert s2_split.train_size == 0
    assert s2_split.test_size == 5


def test_subject_split_is_reproducible_with_seed():
    subjects = [make_subject(f"S{i}", trials=[make_trial(filename=f"s{i}.mat")]) for i in range(10)]

    result_a = subject_split(subjects, n_test_subjects=3, seed=123)
    result_b = subject_split(subjects, n_test_subjects=3, seed=123)

    ids_a = {t.subject_id for t in result_a.trials if t.test_size > 0}
    ids_b = {t.subject_id for t in result_b.trials if t.test_size > 0}

    assert ids_a == ids_b
