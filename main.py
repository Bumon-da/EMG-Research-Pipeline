from config.logging_config import logger

from src.data.loader import DatasetLoader

logger.info("Starting EMG Research Pipeline")

loader = DatasetLoader()

subjects = loader.load()

print()

print("=" * 50)

print(f"Subjects Loaded : {len(subjects)}")

print(f"First Subject : {subjects[0].subject_id}")

print(f"Trials : {subjects[0].num_trials}")

print()

trial = subjects[0].trials[0]

print(trial.filename)

print(trial.samples)

print(trial.channels)