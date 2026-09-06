"""
Dataset Manager
"""

from __future__ import annotations

from src.data.datamodels import Subject
from src.data.loader import DatasetLoader


class DatasetManager:

    def __init__(self) -> None:
        self.loader = DatasetLoader()
        self.subjects: list[Subject] | None = None

    def load(self) -> list[Subject]:
        if self.subjects is None:
            self.subjects = self.loader.load()

        return self.subjects

    def reload(self) -> list[Subject]:
        self.subjects = self.loader.load()
        return self.subjects

    @property
    def subject_count(self) -> int:
        if self.subjects is None:
            return 0

        return len(self.subjects)

    @property
    def trial_count(self) -> int:
        if self.subjects is None:
            return 0

        return sum(len(subject.trials) for subject in self.subjects)
