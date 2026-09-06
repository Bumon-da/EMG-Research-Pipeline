import numpy as np

from src.eda.statistics import Statistics


def test_descriptive_statistics_known_values():
    # Two channels, hand-picked so mean/median/std are easy to verify.
    emg = np.array(
        [
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
            [4.0, 10.0],
        ]
    )

    stats = Statistics.descriptive_statistics(emg)

    assert list(stats.index) == ["Ch1", "Ch2"]

    assert stats.loc["Ch1", "Mean"] == 2.5
    assert stats.loc["Ch1", "Median"] == 2.5
    assert stats.loc["Ch1", "Minimum"] == 1.0
    assert stats.loc["Ch1", "Maximum"] == 4.0

    # Channel 2 is constant: std/variance should be exactly zero.
    assert stats.loc["Ch2", "Std"] == 0.0
    assert stats.loc["Ch2", "Variance"] == 0.0
    assert stats.loc["Ch2", "Mean"] == 10.0

    assert stats.loc["Ch1", "Missing"] == 0


def test_descriptive_statistics_counts_missing():
    emg = np.array(
        [
            [1.0],
            [np.nan],
            [3.0],
        ]
    )

    stats = Statistics.descriptive_statistics(emg)

    assert stats.loc["Ch1", "Missing"] == 1
