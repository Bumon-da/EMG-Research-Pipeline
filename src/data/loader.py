from pathlib import Path
from scipy.io import loadmat

from tqdm import tqdm

from config.settings import RAW_DATA_DIR
from config.logging_config import logger

from src.data.datamodels import Subject, Trial


class DatasetLoader:

    REQUIRED_KEYS = [
        "emg",
        "stimulus",
        "repetition",
        "exercise"
    ]

    def __init__(self):

        self.dataset_path = RAW_DATA_DIR

    def load(self):

        logger.info("Loading dataset...")

        subjects = []

        subject_dirs = sorted([
            d for d in self.dataset_path.iterdir()
            if d.is_dir()
        ])

        logger.info(f"Found {len(subject_dirs)} subjects")

        for subject_dir in tqdm(subject_dirs):

            subject = Subject(
                subject_id=subject_dir.name.upper()
            )

            mat_files = sorted(subject_dir.glob("*.mat"))

            for mat in mat_files:

                data = loadmat(mat)

                missing = [
                    k
                    for k in self.REQUIRED_KEYS
                    if k not in data
                ]

                if missing:

                    logger.warning(
                        f"{mat.name} missing {missing}"
                    )

                    continue

                emg = data["emg"]

                trial = Trial(
                    filename=mat.name,
                    filepath=str(mat),

                    emg=emg,
                    stimulus=data["stimulus"],
                    repetition=data["repetition"],
                    exercise=data["exercise"],

                    samples=emg.shape[0],
                    channels=emg.shape[1]
                )

                subject.trials.append(trial)

            subjects.append(subject)

        logger.info(
            f"Loaded {len(subjects)} subjects successfully."
        )

        return subjects