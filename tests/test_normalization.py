import numpy as np

from src.preprocessing.normalization import Normalizer


def test_clean_clips_negative_values_to_zero():
    emg = np.array([[1.0, -2.0], [3.0, 4.0]])

    clean, negative_count, non_finite_count = Normalizer.clean(emg)

    assert negative_count == 1
    assert non_finite_count == 0
    assert clean[0, 1] == 0.0
    assert clean[0, 0] == 1.0
    assert clean[1, 1] == 4.0


def test_clean_neutralizes_non_finite_values():
    emg = np.array([[1.0, np.nan], [np.inf, 4.0]])

    clean, negative_count, non_finite_count = Normalizer.clean(emg)

    assert non_finite_count == 2
    assert negative_count == 0
    assert clean[0, 1] == 0.0
    assert clean[1, 0] == 0.0


def test_clean_returns_same_object_when_no_negatives_or_non_finite():
    emg = np.array([[1.0, 2.0], [3.0, 4.0]])

    clean, negative_count, non_finite_count = Normalizer.clean(emg)

    assert negative_count == 0
    assert non_finite_count == 0
    assert clean is emg


def test_compute_subject_stats_matches_manual_mean_std():
    emg = np.array([[1.0], [3.0], [5.0], [7.0]])
    train_mask = np.array([True, True, True, True])

    stats = Normalizer.compute_subject_stats("S1", [(emg, train_mask)])

    assert np.isclose(stats.channel_mean[0], emg.mean())
    assert np.isclose(stats.channel_std[0], emg.std())  # ddof=0, matches Normalizer
    assert stats.train_sample_count == 4


def test_compute_subject_stats_excludes_test_rows_from_statistics():
    # Train rows are small, known values; test rows are wildly different.
    # If test rows leaked into the stats, mean/std would be far higher.
    train_rows = np.array([[1.0], [2.0], [3.0], [4.0]])
    test_rows = np.array([[9999.0], [9999.0]])
    emg = np.concatenate([train_rows, test_rows])
    train_mask = np.array([True, True, True, True, False, False])

    stats = Normalizer.compute_subject_stats("S1", [(emg, train_mask)])

    assert np.isclose(stats.channel_mean[0], train_rows.mean())
    assert np.isclose(stats.channel_std[0], train_rows.std())
    assert stats.train_sample_count == 4
    assert not np.isclose(stats.channel_mean[0], emg.mean())


def test_apply_normalizes_test_rows_using_train_derived_stats_not_their_own():
    train_rows = np.array([[0.0], [2.0], [4.0]])  # mean=2
    test_rows = np.array([[100.0], [102.0], [104.0]])  # mean=102
    emg = np.concatenate([train_rows, test_rows])
    train_mask = np.array([True, True, True, False, False, False])

    stats = Normalizer.compute_subject_stats("S1", [(emg, train_mask)])
    result = Normalizer.apply(test_rows, stats)
    expected = (test_rows - stats.channel_mean) / stats.channel_std

    assert np.allclose(result, expected)
    # Prove it's the train mean, not the test rows' own mean, that was used.
    assert np.isclose(stats.channel_mean[0], train_rows.mean())
    assert not np.isclose(stats.channel_mean[0], test_rows.mean())


def test_compute_subject_stats_falls_back_when_zero_train_samples():
    emg = np.array([[1.0], [2.0], [3.0]])
    train_mask = np.array([False, False, False])

    stats = Normalizer.compute_subject_stats("S1", [(emg, train_mask)])

    assert stats.used_test_fallback is True
    assert np.isclose(stats.channel_mean[0], emg.mean())
    assert stats.train_sample_count == 3


def test_compute_subject_stats_floors_zero_variance_channel():
    emg = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])  # channel 0 constant
    train_mask = np.array([True, True, True])

    stats = Normalizer.compute_subject_stats("S1", [(emg, train_mask)])

    assert stats.channel_std[0] == 1.0
    assert 0 in stats.zero_std_channels
    assert np.isclose(stats.channel_mean[0], 1.0)


def test_apply_reproduces_zero_mean_unit_variance_on_its_own_training_distribution():
    rng = np.random.default_rng(0)
    emg = rng.random((100, 3)) + 1.0  # avoid zero variance / negatives
    train_mask = np.ones(100, dtype=bool)

    stats = Normalizer.compute_subject_stats("S1", [(emg, train_mask)])
    normalized = Normalizer.apply(emg, stats)

    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-10)
