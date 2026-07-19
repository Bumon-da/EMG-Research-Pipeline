from dataclasses import dataclass, field
import numpy as np


@dataclass
class Trial:
    filename: str
    filepath: str

    emg: np.ndarray
    stimulus: np.ndarray
    repetition: np.ndarray
    exercise: np.ndarray

    samples: int
    channels: int


@dataclass
class Subject:
    subject_id: str
    trials: list[Trial] = field(default_factory=list)

    @property
    def num_trials(self):
        return len(self.trials)