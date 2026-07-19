"""
Dataset Manager
"""

from src.data.loader import DatasetLoader


class DatasetManager:

    def __init__(self):

        self.loader = DatasetLoader()

        self.subjects = None

    def load(self):

        if self.subjects is None:

            self.subjects = self.loader.load()

        return self.subjects

    def reload(self):

        self.subjects = self.loader.load()

        return self.subjects

    @property
    def subject_count(self):

        if self.subjects is None:
            return 0

        return len(self.subjects)

    @property
    def trial_count(self):

        if self.subjects is None:
            return 0

        return sum(
            len(subject.trials)
            for subject in self.subjects
        )