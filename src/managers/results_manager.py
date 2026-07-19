"""
Results Manager

Centralizes all saving operations.
"""

from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from config.settings import OUTPUT_DIR
from config.logging_config import logger


class ResultsManager:

    def __init__(self):

        self.output_root = Path(OUTPUT_DIR)

        self.output_root.mkdir(
            parents=True,
            exist_ok=True
        )

    def get_folder(self, folder: str) -> Path:

        destination = self.output_root / folder

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        return destination

    def save_csv(
        self,
        dataframe: pd.DataFrame,
        folder: str,
        filename: str,
        index: bool = False
    ) -> Path:

        filepath = self.get_folder(folder) / filename

        dataframe.to_csv(
            filepath,
            index=index
        )

        logger.info(f"CSV Saved -> {filepath}")

        return filepath

    def save_plot(
        self,
        figure,
        folder: str,
        filename: str,
        dpi: int = 300
    ) -> Path:

        filepath = self.get_folder(folder) / filename

        figure.savefig(
            filepath,
            dpi=dpi,
            bbox_inches="tight"
        )

        plt.close(figure)

        logger.info(f"Plot Saved -> {filepath}")

        return filepath

    def save_text(
        self,
        text: str,
        folder: str,
        filename: str
    ) -> Path:

        filepath = self.get_folder(folder) / filename

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(text)

        logger.info(f"Text Saved -> {filepath}")

        return filepath

    def timestamp(self) -> str:

        return datetime.now().strftime("%Y%m%d_%H%M%S")