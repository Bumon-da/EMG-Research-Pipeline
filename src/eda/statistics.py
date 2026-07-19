"""
Statistical functions for EMG datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Statistics:

    @staticmethod
    def descriptive_statistics(emg: np.ndarray) -> pd.DataFrame:
        """
        Generate descriptive statistics for every EMG channel.
        """

        df = pd.DataFrame(
            emg,
            columns=[f"Ch{i+1}" for i in range(emg.shape[1])]
        )

        stats = pd.DataFrame({
            "Mean": df.mean(),
            "Median": df.median(),
            "Std": df.std(),
            "Variance": df.var(),
            "Minimum": df.min(),
            "Maximum": df.max(),
            "Q1": df.quantile(0.25),
            "Q3": df.quantile(0.75),
            "Missing": df.isna().sum(),
        })

        return stats