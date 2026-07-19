"""
Experiment Manager
"""

from pathlib import Path
from datetime import datetime

from config.settings import OUTPUT_DIR


class ExperimentManager:

    def __init__(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_name = f"experiment_{timestamp}"

        self.path = (
            Path(OUTPUT_DIR)
            / "experiments"
            / self.experiment_name
        )

        self.path.mkdir(
            parents=True,
            exist_ok=True
        )

    def folder(self, name: str):

        directory = self.path / name

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        return directory