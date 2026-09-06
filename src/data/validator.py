"""
Dataset Validation

Performs both basic integrity checks (missing/NaN/Inf/empty/constant
channels) and advanced, research-oriented checks (duplicate trials, signal
range/saturation, per-channel electrode-contact quality, trial-duration
consistency, and gesture class balance) on the loaded EMG dataset.

Note on NinaPro DB1 specifically: its "emg" field is not raw broadband
sEMG. The Otto Bock 13E200 electrodes used to record DB1 output an
already-rectified, non-negative, quantized envelope signal (confirmed by
inspection: values are >= 0 and step in ~0.0024 increments, consistent
with the sensor's onboard ADC) rather than a raw AC-coupled waveform. That
is why `negative_values` is treated as an integrity problem below, and why
a bandpass/notch filter tuned for raw sEMG (see config.settings) should
not be applied to this signal unchanged - that decision belongs to the
preprocessing stage, but the validator flags the discrepancy here because
it is a dataset property, not a preprocessing choice.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from config.logging_config import logger
from config.settings import SAMPLING_RATE
from src.data.datamodels import Subject, Trial
from src.managers.results_manager import ResultsManager


@dataclass
class ValidationResult:
    subject: str
    trial: str
    exercise: int | None
    samples: int
    channels: int
    duration_sec: float
    mean_rms: float

    missing_values: int
    nan_values: int
    infinite_values: int
    negative_values: int
    empty_channels: int
    constant_channels: int
    saturated_channels: int
    low_activity_channels: int

    is_duplicate: bool
    duration_outlier: bool
    has_refined_labels: bool

    flagged: bool


class DatasetValidator:

    # A channel is flagged "saturated" if more than this fraction of its
    # samples sit at (or within 0.1% of) the trial's own observed ceiling
    # for that channel - a heuristic for ADC clipping.
    SATURATION_FRACTION_THRESHOLD = 0.01

    # A channel is flagged "low activity" (possible poor electrode contact)
    # if its RMS is below this fraction of the trial's median channel RMS.
    LOW_ACTIVITY_RMS_RATIO = 0.05

    # A trial is flagged as a duration outlier if it deviates from the
    # median duration of trials sharing the same exercise by more than
    # this fraction.
    DURATION_OUTLIER_TOLERANCE = 0.25

    # Floating-point std() of a constant array is not always exactly 0.0
    # (mean-then-subtract rounding leaves residues around 1e-16), so
    # "constant channel" is a tolerance check, not an exact-equality one.
    CONSTANT_CHANNEL_STD_EPS = 1e-9

    def __init__(self, results: ResultsManager | None = None) -> None:
        self.results = results or ResultsManager()

    def validate(self, subjects: list[Subject]) -> pd.DataFrame:
        logger.info("Starting Dataset Validation...")

        median_duration = self._median_duration_by_exercise(subjects)

        rows: list[ValidationResult] = []
        seen_hashes: dict[str, tuple[str, str]] = {}

        for subject in subjects:
            for trial in subject.trials:
                rows.append(
                    self._validate_trial(
                        subject.subject_id, trial, median_duration, seen_hashes
                    )
                )

        details_df = pd.DataFrame([asdict(r) for r in rows])

        self.results.save_dataframe(
            dataframe=details_df,
            folder="validation",
            filename="validation_details.csv",
        )

        flagged_df = details_df[details_df["flagged"]]
        self.results.save_dataframe(
            dataframe=flagged_df,
            folder="validation",
            filename="flagged_trials.csv",
        )

        class_distribution_df = self._class_distribution(subjects)
        self.results.save_dataframe(
            dataframe=class_distribution_df,
            folder="validation",
            filename="class_distribution.csv",
        )

        summary_df = self._summary(details_df, class_distribution_df)
        self.results.save_dataframe(
            dataframe=summary_df,
            folder="validation",
            filename="validation_summary.csv",
        )

        report_text = self._integrity_report(details_df, flagged_df, class_distribution_df)
        self.results.save_text(
            text=report_text,
            folder="validation",
            filename="integrity_report.txt",
        )

        print("\n")
        print("=" * 60)
        print("DATASET VALIDATION")
        print("=" * 60)

        for _, row in summary_df.iterrows():
            print(f"{row['Metric']:<38}: {row['Value']}")

        logger.info("Dataset Validation Complete.")

        return details_df

    # ------------------------------------------------------------------
    # Per-trial validation
    # ------------------------------------------------------------------

    def _validate_trial(
        self,
        subject_id: str,
        trial: Trial,
        median_duration: dict[str, float],
        seen_hashes: dict[str, tuple[str, str]],
    ) -> ValidationResult:

        emg = trial.emg
        duration_sec = trial.samples / SAMPLING_RATE

        missing = int(pd.isnull(emg).sum())
        nan = int(np.isnan(emg).sum())
        infinite = int(np.isinf(emg).sum())
        negative = int(np.sum(emg < 0))

        empty = int(np.sum(np.all(emg == 0, axis=0)))
        constant = int(np.sum(np.std(emg, axis=0) <= self.CONSTANT_CHANNEL_STD_EPS))

        saturated = self._count_saturated_channels(emg)
        low_activity = self._count_low_activity_channels(emg)

        channel_rms = np.sqrt(np.mean(np.square(emg), axis=0))
        mean_rms = float(np.mean(channel_rms))

        emg_hash = hashlib.sha1(np.ascontiguousarray(emg).tobytes()).hexdigest()
        is_duplicate = emg_hash in seen_hashes
        if not is_duplicate:
            seen_hashes[emg_hash] = (subject_id, trial.filename)
        else:
            other_subject, other_trial = seen_hashes[emg_hash]
            logger.warning(
                f"{subject_id}/{trial.filename} is byte-identical to "
                f"{other_subject}/{other_trial}"
            )

        exercise_key = self._exercise_key(trial)
        expected_duration = median_duration.get(exercise_key)
        duration_outlier = bool(
            expected_duration
            and expected_duration > 0
            and abs(duration_sec - expected_duration) / expected_duration
            > self.DURATION_OUTLIER_TOLERANCE
        )

        flagged = bool(
            missing or nan or infinite or negative
            or empty or constant or saturated or low_activity
            or is_duplicate or duration_outlier
        )

        return ValidationResult(
            subject=subject_id,
            trial=trial.filename,
            exercise=trial.exercise_id,
            samples=trial.samples,
            channels=trial.channels,
            duration_sec=round(duration_sec, 2),
            mean_rms=round(mean_rms, 6),
            missing_values=missing,
            nan_values=nan,
            infinite_values=infinite,
            negative_values=negative,
            empty_channels=empty,
            constant_channels=constant,
            saturated_channels=saturated,
            low_activity_channels=low_activity,
            is_duplicate=is_duplicate,
            duration_outlier=duration_outlier,
            has_refined_labels=trial.has_refined_labels,
            flagged=flagged,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _exercise_key(trial: Trial) -> str:
        exercise_id = trial.exercise_id
        return str(exercise_id) if exercise_id is not None else "unknown"

    def _median_duration_by_exercise(self, subjects: list[Subject]) -> dict[str, float]:
        durations: dict[str, list[float]] = {}

        for subject in subjects:
            for trial in subject.trials:
                key = self._exercise_key(trial)
                durations.setdefault(key, []).append(trial.samples / SAMPLING_RATE)

        return {key: float(np.median(values)) for key, values in durations.items()}

    # A single sample always equals the column max, so for small trials
    # "fraction at ceiling" alone would flag on statistical noise (e.g.
    # 1/50 = 2%). Requiring a minimum absolute count as well means the
    # check only fires when many samples are genuinely pinned at the same
    # value - the actual signature of ADC clipping on a quantized signal.
    MIN_SATURATED_SAMPLES = 5

    def _count_saturated_channels(self, emg: np.ndarray) -> int:
        count = 0

        for ch in range(emg.shape[1]):
            column = emg[:, ch]
            ceiling = column.max()

            if ceiling <= 0:
                continue

            samples_at_ceiling = int(np.sum(column >= ceiling * 0.999))
            fraction_at_ceiling = samples_at_ceiling / column.shape[0]

            if (
                samples_at_ceiling >= self.MIN_SATURATED_SAMPLES
                and fraction_at_ceiling > self.SATURATION_FRACTION_THRESHOLD
            ):
                count += 1

        return count

    def _count_low_activity_channels(self, emg: np.ndarray) -> int:
        rms = np.sqrt(np.mean(np.square(emg), axis=0))
        median_rms = np.median(rms)

        if median_rms <= 0:
            return 0

        return int(np.sum(rms < median_rms * self.LOW_ACTIVITY_RMS_RATIO))

    @staticmethod
    def _class_distribution(subjects: list[Subject]) -> pd.DataFrame:
        """
        Sample count per (exercise, gesture label). Grouping must include
        exercise: NinaPro's label numbering resets at 0 for every
        exercise, so "label 5" means a different gesture in exercise 1
        than in exercise 2 - summing across exercises by label alone
        would silently merge unrelated gestures.
        """

        rows = []

        for subject in subjects:
            for trial in subject.trials:
                exercise_id = trial.exercise_id
                for label, count in trial.gesture_counts.items():
                    rows.append(
                        {
                            "Subject": subject.subject_id,
                            "Trial": trial.filename,
                            "Exercise": exercise_id,
                            "Label": label,
                            "Samples": count,
                        }
                    )

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        return (
            df.groupby(["Exercise", "Label"], as_index=False)["Samples"]
            .sum()
            .sort_values(["Exercise", "Label"])
            .reset_index(drop=True)
        )

    @staticmethod
    def _summary(details_df: pd.DataFrame, class_distribution_df: pd.DataFrame) -> pd.DataFrame:
        # Imbalance is computed within each exercise separately (labels
        # are not comparable across exercises - see _class_distribution)
        # and reported as the worst ratio seen in any single exercise.
        if not class_distribution_df.empty:
            gesture_only = class_distribution_df[class_distribution_df["Label"] != 0]
            ratios = []
            for _, group in gesture_only.groupby("Exercise"):
                if not group.empty:
                    ratios.append(group["Samples"].max() / max(group["Samples"].min(), 1))
            imbalance_ratio = round(max(ratios), 2) if ratios else float("nan")
        else:
            imbalance_ratio = float("nan")

        metrics = {
            "Subjects": details_df["subject"].nunique(),
            "Trials": len(details_df),
            "Missing Values": int(details_df["missing_values"].sum()),
            "NaN Values": int(details_df["nan_values"].sum()),
            "Infinite Values": int(details_df["infinite_values"].sum()),
            "Negative Values (unexpected for DB1)": int(details_df["negative_values"].sum()),
            "Empty Channels": int(details_df["empty_channels"].sum()),
            "Constant Channels": int(details_df["constant_channels"].sum()),
            "Saturated Channels": int(details_df["saturated_channels"].sum()),
            "Low-Activity Channels": int(details_df["low_activity_channels"].sum()),
            "Duplicate Trials": int(details_df["is_duplicate"].sum()),
            "Duration Outliers": int(details_df["duration_outlier"].sum()),
            "Trials Missing Refined Labels": int((~details_df["has_refined_labels"]).sum()),
            "Flagged Trials": int(details_df["flagged"].sum()),
            "Gesture Class Imbalance Ratio (max/min, worst exercise)": imbalance_ratio,
        }

        return pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})

    @staticmethod
    def _integrity_report(
        details_df: pd.DataFrame,
        flagged_df: pd.DataFrame,
        class_distribution_df: pd.DataFrame,
    ) -> str:

        lines = [
            "EMG DATASET INTEGRITY REPORT",
            "=" * 60,
            f"Total trials validated : {len(details_df)}",
            f"Flagged trials         : {len(flagged_df)}",
            "",
            "Flagged trials (see flagged_trials.csv for full detail):",
        ]

        if flagged_df.empty:
            lines.append("  None.")
        else:
            for _, row in flagged_df.iterrows():
                reasons = []
                if row["missing_values"]:
                    reasons.append("missing values")
                if row["nan_values"]:
                    reasons.append("NaN values")
                if row["infinite_values"]:
                    reasons.append("infinite values")
                if row["negative_values"]:
                    reasons.append("unexpected negative values")
                if row["empty_channels"]:
                    reasons.append("empty channel(s)")
                if row["constant_channels"]:
                    reasons.append("constant channel(s)")
                if row["saturated_channels"]:
                    reasons.append("possible saturation")
                if row["low_activity_channels"]:
                    reasons.append("possible poor electrode contact")
                if row["is_duplicate"]:
                    reasons.append("duplicate of another trial")
                if row["duration_outlier"]:
                    reasons.append("unusual trial duration")

                lines.append(f"  - {row['subject']}/{row['trial']}: {', '.join(reasons)}")

        lines.append("")
        lines.append(
            "Gesture class distribution (sample counts, label 0 = rest; "
            "labels are exercise-relative, not comparable across exercises):"
        )

        if class_distribution_df.empty:
            lines.append("  (no label data)")
        else:
            for exercise, group in class_distribution_df.groupby("Exercise"):
                lines.append(f"  Exercise {exercise}:")
                for _, row in group.iterrows():
                    lines.append(
                        f"    Label {int(row['Label']):>2} : {int(row['Samples']):>10,} samples"
                    )

        return "\n".join(lines)
