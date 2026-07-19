"""
Dataset Validation

Performs integrity checks on the EMG dataset.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.logging_config import logger
from src.managers.results_manager import ResultsManager


@dataclass
class ValidationResult:
    subject: str
    trial: str
    samples: int
    channels: int
    missing_values: int
    nan_values: int
    infinite_values: int
    empty_channels: int
    constant_channels: int


class DatasetValidator:

    def __init__(self):

        self.results = ResultsManager()

    def validate(self, subjects):

        logger.info("Starting Dataset Validation...")

        validation_results = []

        for subject in subjects:

            for trial in subject.trials:

                emg = trial.emg

                missing = int(pd.isnull(emg).sum())

                nan = int(np.isnan(emg).sum())

                infinite = int(np.isinf(emg).sum())

                empty = int(np.sum(np.all(emg == 0, axis=0)))

                constant = int(np.sum(np.std(emg, axis=0) == 0))

                validation_results.append(
                    ValidationResult(
                        subject.subject_id,
                        trial.filename,
                        trial.samples,
                        trial.channels,
                        missing,
                        nan,
                        infinite,
                        empty,
                        constant,
                    )
                )

        df = pd.DataFrame([vars(r) for r in validation_results])

        self.results.save_dataframe(
            dataframe=df,
            folder="validation",
            filename="validation_details.csv",
        )

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Subjects",
                    "Trials",
                    "Missing Values",
                    "NaN Values",
                    "Infinite Values",
                    "Empty Channels",
                    "Constant Channels",
                ],
                "Value": [
                    len(subjects),
                    len(df),
                    df["missing_values"].sum(),
                    df["nan_values"].sum(),
                    df["infinite_values"].sum(),
                    df["empty_channels"].sum(),
                    df["constant_channels"].sum(),
                ],
            }
        )

        self.results.save_dataframe(
            dataframe=summary,
            folder="validation",
            filename="validation_summary.csv",
        )

        print("\n")
        print("=" * 60)
        print("DATASET VALIDATION")
        print("=" * 60)

        for _, row in summary.iterrows():
            print(f"{row['Metric']:<22}: {row['Value']}")

        logger.info("Dataset Validation Complete.")