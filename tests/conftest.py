"""
Shared synthetic-data helpers for tests. None of these tests touch the
real dataset - everything is built from small numpy arrays so the suite
runs anywhere, fast.
"""

from __future__ import annotations

import numpy as np

from src.data.datamodels import Subject, Trial


def make_trial(
    filename: str = "S1_A1_E1.mat",
    n_samples: int = 20,
    n_channels: int = 4,
    stimulus: np.ndarray | None = None,
    repetition: np.ndarray | None = None,
    restimulus: np.ndarray | None = None,
    rerepetition: np.ndarray | None = None,
    emg: np.ndarray | None = None,
    glove: np.ndarray | None = None,
) -> Trial:
    rng = np.random.default_rng(0)

    emg = emg if emg is not None else rng.random((n_samples, n_channels))
    stimulus = stimulus if stimulus is not None else np.zeros(n_samples, dtype=int)
    repetition = repetition if repetition is not None else np.zeros(n_samples, dtype=int)

    return Trial(
        filename=filename,
        filepath=f"/fake/{filename}",
        emg=emg,
        stimulus=stimulus,
        repetition=repetition,
        exercise=np.array([[1]]),
        samples=emg.shape[0],
        channels=emg.shape[1],
        restimulus=restimulus,
        rerepetition=rerepetition,
        glove=glove,
    )


def make_subject(subject_id: str = "S1", trials: list[Trial] | None = None) -> Subject:
    return Subject(subject_id=subject_id, trials=trials if trials is not None else [make_trial()])
