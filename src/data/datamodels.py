"""
Core data structures for the EMG Research Pipeline.

A `Trial` wraps a single NinaPro-style `.mat` recording. Both the raw
stimulus/repetition labels and the *refined* restimulus/rerepetition labels
are kept, because they are not interchangeable:

- `stimulus` / `repetition` are stamped from a fixed reaction-time offset
  and do not align with true muscle-activation onset.
- `restimulus` / `rerepetition` are NinaPro's corrected labels and are the
  ones almost every published gesture-recognition result is trained on.

Downstream code should default to `restimulus`/`rerepetition` (see
`Trial.labels` / `Trial.reps`) unless there is a specific reason to use the
raw fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trial:
    """
    A single recorded exercise trial for one subject.

    Attributes:
        filename: Source `.mat` filename (e.g. "S1_A1_E1.mat").
        filepath: Absolute path to the source file.
        emg: (samples, channels) surface EMG signal.
        glove: (samples, 22) Cyberglove joint-angle signal, or None if the
            file did not contain one (not all NinaPro subsets record it).
        stimulus: (samples,) raw gesture label per sample.
        restimulus: (samples,) refined/corrected gesture label per sample.
            Prefer this over `stimulus` for classification work.
        repetition: (samples,) raw repetition index per sample.
        rerepetition: (samples,) refined repetition index per sample.
            Prefer this over `repetition` for train/test splitting.
        exercise: NinaPro exercise letter/number this file covers (A/B/C).
        samples: Number of time samples in the trial (== emg.shape[0]).
        channels: Number of EMG channels (== emg.shape[1]).
    """

    filename: str
    filepath: str

    emg: np.ndarray
    stimulus: np.ndarray
    repetition: np.ndarray
    exercise: np.ndarray

    samples: int
    channels: int

    restimulus: np.ndarray | None = None
    rerepetition: np.ndarray | None = None
    glove: np.ndarray | None = None

    @property
    def labels(self) -> np.ndarray:
        """
        Per-sample gesture label, preferring the refined `restimulus` when
        available and falling back to raw `stimulus` otherwise.
        """
        return self.restimulus if self.restimulus is not None else self.stimulus

    @property
    def reps(self) -> np.ndarray:
        """
        Per-sample repetition index, preferring the refined `rerepetition`
        when available and falling back to raw `repetition` otherwise.
        """
        return self.rerepetition if self.rerepetition is not None else self.repetition

    @property
    def has_refined_labels(self) -> bool:
        return self.restimulus is not None and self.rerepetition is not None

    @property
    def has_glove(self) -> bool:
        return self.glove is not None

    @property
    def exercise_id(self) -> int | None:
        """
        The NinaPro exercise number this trial belongs to (1, 2, 3, ...).

        Important: gesture label numbering resets per exercise - label 5
        in exercise 1 is not the same gesture as label 5 in exercise 2.
        Any aggregation across trials (class distribution, plots, splits)
        must key on (exercise_id, label), never on label alone.
        """
        try:
            return int(np.asarray(self.exercise).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return None

    @property
    def gesture_counts(self) -> dict[int, int]:
        """
        Sample count per gesture label (using the preferred label source).
        Label 0 is NinaPro's "rest" class.
        """
        labels = np.asarray(self.labels).reshape(-1)
        values, counts = np.unique(labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}


@dataclass
class Subject:
    subject_id: str
    trials: list[Trial] = field(default_factory=list)

    @property
    def num_trials(self) -> int:
        return len(self.trials)

    @property
    def total_samples(self) -> int:
        return sum(trial.samples for trial in self.trials)
