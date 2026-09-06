"""
EDA Visualization

Produces the figures a paper's "Dataset" section typically needs: raw
signal traces, per-channel distribution plots, a channel-correlation
heatmap, a cross-subject comparison, and the gesture class distribution.

All figures are saved through `ResultsManager.save_plot` (never shown
interactively), keeping this class usable from a headless pipeline run.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.logging_config import logger
from config.settings import SAMPLING_RATE
from src.data.datamodels import Subject, Trial
from src.managers.results_manager import ResultsManager


class Visualizer:

    def __init__(self, results: ResultsManager | None = None) -> None:
        self.results = results or ResultsManager()
        self.folder = "eda/figures"

    # ------------------------------------------------------------------
    # Raw signal
    # ------------------------------------------------------------------

    def plot_raw_signal(
        self,
        trial: Trial,
        max_seconds: float = 5.0,
        max_channels: int = 5,
    ) -> None:
        """
        Plot the first `max_seconds` of the first `max_channels` channels
        of a trial, one subplot per channel, so signal character can be
        inspected visually.
        """

        n_samples = min(trial.samples, int(max_seconds * SAMPLING_RATE))
        n_channels = min(trial.channels, max_channels)
        time_axis = np.arange(n_samples) / SAMPLING_RATE

        fig, axes = plt.subplots(
            n_channels, 1, figsize=(10, 1.6 * n_channels), sharex=True
        )
        if n_channels == 1:
            axes = [axes]

        for ch in range(n_channels):
            axes[ch].plot(time_axis, trial.emg[:n_samples, ch], linewidth=0.7)
            axes[ch].set_ylabel(f"Ch{ch + 1}")
            axes[ch].grid(alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Raw EMG - {trial.filename}")
        fig.tight_layout()

        self.results.save_plot(
            fig, folder=self.folder, filename=f"raw_signal_{trial.filename}.png"
        )

    # ------------------------------------------------------------------
    # Distributions
    # ------------------------------------------------------------------

    def plot_channel_histograms(self, trial: Trial, bins: int = 60) -> None:
        fig, axes = plt.subplots(
            2, (trial.channels + 1) // 2, figsize=(3 * ((trial.channels + 1) // 2), 6)
        )
        axes = np.asarray(axes).reshape(-1)

        for ch in range(trial.channels):
            axes[ch].hist(trial.emg[:, ch], bins=bins, color="#4C72B0")
            axes[ch].set_title(f"Ch{ch + 1}")

        for ax in axes[trial.channels:]:
            ax.axis("off")

        fig.suptitle(f"Channel Amplitude Histograms - {trial.filename}")
        fig.tight_layout()

        self.results.save_plot(
            fig, folder=self.folder, filename=f"histograms_{trial.filename}.png"
        )

    def plot_channel_boxplots(self, trial: Trial) -> None:
        fig, ax = plt.subplots(figsize=(1.1 * trial.channels + 2, 5))

        ax.boxplot(
            [trial.emg[:, ch] for ch in range(trial.channels)],
            tick_labels=[f"Ch{ch + 1}" for ch in range(trial.channels)],
            showfliers=False,
        )
        ax.set_title(f"Channel Amplitude Spread - {trial.filename}")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3, axis="y")

        self.results.save_plot(
            fig, folder=self.folder, filename=f"boxplot_{trial.filename}.png"
        )

    def plot_channel_correlation_heatmap(self, trial: Trial) -> None:
        corr = np.corrcoef(trial.emg, rowvar=False)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")

        ax.set_xticks(range(trial.channels))
        ax.set_yticks(range(trial.channels))
        ax.set_xticklabels([f"Ch{ch + 1}" for ch in range(trial.channels)], rotation=90)
        ax.set_yticklabels([f"Ch{ch + 1}" for ch in range(trial.channels)])
        ax.set_title(f"Inter-Channel Correlation - {trial.filename}")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        self.results.save_plot(
            fig, folder=self.folder, filename=f"correlation_{trial.filename}.png"
        )

    # ------------------------------------------------------------------
    # Cross-subject / cross-dataset
    # ------------------------------------------------------------------

    def plot_subject_comparison(self, subjects: list[Subject]) -> None:
        """
        Mean per-subject RMS (averaged across channels and trials), to
        spot subjects whose overall signal amplitude is unusually low or
        high relative to the rest of the cohort.
        """

        rows = []
        for subject in subjects:
            trial_rms = []
            for trial in subject.trials:
                rms = np.sqrt(np.mean(np.square(trial.emg)))
                trial_rms.append(rms)
            if trial_rms:
                rows.append({"Subject": subject.subject_id, "MeanRMS": float(np.mean(trial_rms))})

        if not rows:
            logger.warning("No subject data available for subject comparison plot.")
            return

        df = pd.DataFrame(rows).sort_values("Subject")

        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(df)), 5))
        ax.bar(df["Subject"], df["MeanRMS"], color="#55A868")
        ax.set_ylabel("Mean EMG RMS")
        ax.set_title("Mean Signal Amplitude by Subject")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()

        self.results.save_plot(fig, folder=self.folder, filename="subject_comparison.png")

    def plot_gesture_distribution(self, subjects: list[Subject]) -> None:
        """
        Sample count per gesture label, faceted by exercise (using the
        refined label when available). Label 0 is rest.

        NinaPro's gesture-label numbering resets at 0 for every exercise
        (label 5 in exercise 1 is a different gesture than label 5 in
        exercise 2), so this deliberately produces one subplot per
        exercise rather than a single merged bar chart, which would
        silently combine unrelated gestures under the same x position.
        """

        # counts[exercise_id][label] = sample count
        counts: dict[int, dict[int, int]] = {}

        for subject in subjects:
            for trial in subject.trials:
                exercise_id = trial.exercise_id if trial.exercise_id is not None else -1
                exercise_counts = counts.setdefault(exercise_id, {})
                for label, n in trial.gesture_counts.items():
                    exercise_counts[label] = exercise_counts.get(label, 0) + n

        if not counts:
            logger.warning("No label data available for gesture distribution plot.")
            return

        exercises = sorted(counts)
        fig, axes = plt.subplots(1, len(exercises), figsize=(6 * len(exercises), 5), squeeze=False)
        axes = axes[0]

        for ax, exercise_id in zip(axes, exercises):
            labels = sorted(counts[exercise_id])
            values = [counts[exercise_id][label] for label in labels]
            positions = np.arange(len(labels))

            ax.bar(positions, values, color="#C44E52")
            ax.set_xticks(positions)
            ax.set_xticklabels([str(label) for label in labels])
            ax.set_xlabel("Gesture Label (0 = rest)")
            ax.set_ylabel("Sample Count")
            ax.set_title(f"Exercise {exercise_id}")
            ax.grid(alpha=0.3, axis="y")

        fig.suptitle("Gesture Class Distribution by Exercise")
        fig.tight_layout()

        self.results.save_plot(fig, folder=self.folder, filename="gesture_distribution.png")
