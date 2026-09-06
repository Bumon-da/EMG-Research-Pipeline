"""
Dataset loading for NinaPro-style `.mat` recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from config.logging_config import logger
from config.settings import RAW_DATA_DIR
from src.data.datamodels import Subject, Trial


class DatasetLoader:
    """
    Loads every subject folder under `RAW_DATA_DIR` and parses each
    `.mat` file into a `Trial`.
    """

    # Keys every trial must contain to be usable at all.
    REQUIRED_KEYS = [
        "emg",
        "stimulus",
        "repetition",
        "exercise",
    ]

    # Keys that are used when present but are not fatal if missing
    # (some NinaPro subsets omit refined labels or glove data).
    OPTIONAL_KEYS = [
        "restimulus",
        "rerepetition",
        "glove",
    ]

    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path else RAW_DATA_DIR

    def list_subject_ids(self) -> list[str]:
        """
        Subject IDs available on disk, without parsing any `.mat` file.
        Cheap directory listing - safe to call often (e.g. to populate a
        UI dropdown).
        """

        if not self.dataset_path.exists():
            return []

        return sorted(
            d.name.upper() for d in self.dataset_path.iterdir() if d.is_dir()
        )

    def list_trial_filenames(self, subject_id: str) -> list[str]:
        """
        `.mat` filenames available for one subject, without parsing them.
        """

        subject_dir = self._resolve_subject_dir(subject_id)

        if subject_dir is None:
            return []

        return sorted(p.name for p in subject_dir.glob("*.mat"))

    def load_trial(self, subject_id: str, filename: str) -> Trial | None:
        """
        Load exactly one `.mat` file on demand, without touching the rest
        of the dataset. Intended for interactive use (e.g. a dashboard's
        signal browser) where loading all 27 subjects just to look at one
        trial would be wasteful.
        """

        subject_dir = self._resolve_subject_dir(subject_id)

        if subject_dir is None:
            logger.warning(f"Subject folder not found for '{subject_id}'")
            return None

        mat_path = subject_dir / filename

        if not mat_path.exists():
            logger.warning(f"Trial file not found: {mat_path}")
            return None

        return self._load_trial(mat_path)

    def _resolve_subject_dir(self, subject_id: str) -> Path | None:
        """
        Subject folders on disk may not match `subject_id`'s casing
        (`Trial`/`Subject` normalize IDs to uppercase, folder names may
        not be), so resolve case-insensitively rather than assuming
        `self.dataset_path / subject_id` exists as-is.
        """

        if not self.dataset_path.exists():
            return None

        direct = self.dataset_path / subject_id
        if direct.is_dir():
            return direct

        for d in self.dataset_path.iterdir():
            if d.is_dir() and d.name.upper() == subject_id.upper():
                return d

        return None

    def load(self) -> list[Subject]:
        logger.info("Loading dataset...")

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Raw dataset directory not found: {self.dataset_path}"
            )

        subjects: list[Subject] = []

        subject_dirs = sorted(
            d for d in self.dataset_path.iterdir() if d.is_dir()
        )

        logger.info(f"Found {len(subject_dirs)} subjects")

        for subject_dir in tqdm(subject_dirs):

            subject = Subject(subject_id=subject_dir.name.upper())

            mat_files = sorted(subject_dir.glob("*.mat"))

            for mat in mat_files:

                trial = self._load_trial(mat)

                if trial is not None:
                    subject.trials.append(trial)

            subjects.append(subject)

        logger.info(f"Loaded {len(subjects)} subjects successfully.")

        return subjects

    def _load_trial(self, mat_path: Path) -> Trial | None:
        """
        Parse a single `.mat` file into a `Trial`, or return None (with a
        logged warning) if it is missing required fields or is internally
        inconsistent.
        """

        data = loadmat(mat_path)

        missing = [k for k in self.REQUIRED_KEYS if k not in data]

        if missing:
            logger.warning(f"{mat_path.name} missing required keys {missing} - skipped")
            return None

        emg = data["emg"]
        stimulus = data["stimulus"]
        repetition = data["repetition"]
        exercise = data["exercise"]

        restimulus = data.get("restimulus")
        rerepetition = data.get("rerepetition")
        glove = data.get("glove")

        # Every per-sample array must have the same length as emg, or
        # everything downstream (windowing, labeling, splitting) silently
        # misaligns. This is checked here rather than left to validation
        # because a misaligned trial is not safely usable at all.
        n_samples = emg.shape[0]

        per_sample_arrays = {
            "stimulus": stimulus,
            "repetition": repetition,
        }
        if restimulus is not None:
            per_sample_arrays["restimulus"] = restimulus
        if rerepetition is not None:
            per_sample_arrays["rerepetition"] = rerepetition
        if glove is not None:
            per_sample_arrays["glove"] = glove

        mismatched = [
            name
            for name, arr in per_sample_arrays.items()
            if arr.shape[0] != n_samples
        ]

        if mismatched:
            logger.warning(
                f"{mat_path.name} has length-mismatched arrays {mismatched} "
                f"(emg has {n_samples} samples) - skipped"
            )
            return None

        return Trial(
            filename=mat_path.name,
            filepath=str(mat_path),
            emg=emg,
            stimulus=stimulus,
            repetition=repetition,
            exercise=exercise,
            restimulus=restimulus,
            rerepetition=rerepetition,
            glove=glove,
            samples=n_samples,
            channels=emg.shape[1],
        )
