"""
Results Manager

Centralized manager responsible for saving all outputs generated
throughout the EMG Research Pipeline.

Supported output types:
- CSV files
- Text reports
- Matplotlib figures

Future support:
- Excel workbooks
- JSON files
- HTML reports
- Trained ML models
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.logging_config import logger
from config.settings import OUTPUT_DIR


class ResultsManager:
    """
    Handles creation of output directories and saving of
    files generated during the pipeline.
    """

    def __init__(self) -> None:
        self.output_root = Path(OUTPUT_DIR)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def get_folder(self, folder: str) -> Path:
        """
        Create (if necessary) and return an output folder.

        Example:
            folder="validation"
            folder="eda/statistics"
        """

        destination = self.output_root / folder
        destination.mkdir(parents=True, exist_ok=True)

        return destination

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    def save_csv(
        self,
        dataframe: pd.DataFrame,
        folder: str,
        filename: str,
        index: bool = False,
    ) -> Path:
        """
        Save a DataFrame as a CSV file.
        """

        filepath = self.get_folder(folder) / filename

        dataframe.to_csv(filepath, index=index)

        logger.info(f"CSV saved -> {filepath}")

        return filepath

    def save_dataframe(
        self,
        dataframe: pd.DataFrame,
        folder: str,
        filename: str,
        index: bool = False,
    ) -> Path:
        """
        Alias for save_csv().

        This method exists to make calling code read more naturally.
        """

        return self.save_csv(
            dataframe=dataframe,
            folder=folder,
            filename=filename,
            index=index,
        )

    # ---------------------------------------------------------
    # Text
    # ---------------------------------------------------------

    def save_text(
        self,
        text: str,
        folder: str,
        filename: str,
    ) -> Path:
        """
        Save plain text to a file.
        """

        filepath = self.get_folder(folder) / filename

        filepath.write_text(text, encoding="utf-8")

        logger.info(f"Text report saved -> {filepath}")

        return filepath

    # ---------------------------------------------------------
    # Figures
    # ---------------------------------------------------------

    def save_plot(
        self,
        figure,
        folder: str,
        filename: str,
        dpi: int = 300,
    ) -> Path:
        """
        Save a Matplotlib figure.
        """

        filepath = self.get_folder(folder) / filename

        figure.savefig(
            filepath,
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close(figure)

        logger.info(f"Plot saved -> {filepath}")

        return filepath

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @staticmethod
    def timestamp() -> str:
        """
        Return a timestamp string suitable for filenames.
        """

        return datetime.now().strftime("%Y%m%d_%H%M%S")