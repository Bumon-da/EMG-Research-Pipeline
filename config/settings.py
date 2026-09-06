"""
Global configuration for the EMG Research Pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

# -------------------------------------------------
# Project Paths
# -------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"

# The raw dataset directory is resolved in this order:
#   1. EMG_RAW_DATA_DIR environment variable, if set.
#   2. data/raw/DB1_Extracted (the default location).
_local_raw_dir = DATA_DIR / "raw" / "DB1_Extracted"

if "EMG_RAW_DATA_DIR" in os.environ:
    RAW_DATA_DIR: Path = Path(os.environ["EMG_RAW_DATA_DIR"])
else:
    RAW_DATA_DIR = _local_raw_dir

PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

OUTPUT_DIR: Path = PROJECT_ROOT / "output"

VALIDATION_DIR: Path = OUTPUT_DIR / "validation"
EDA_DIR: Path = OUTPUT_DIR / "eda"
PREPROCESS_DIR: Path = OUTPUT_DIR / "preprocessing"
FEATURE_DIR: Path = OUTPUT_DIR / "features"
MODEL_DIR: Path = OUTPUT_DIR / "models"
REPORT_DIR: Path = OUTPUT_DIR / "reports"
EXPERIMENTS_DIR: Path = OUTPUT_DIR / "experiments"

# -------------------------------------------------
# Dataset
# -------------------------------------------------

NUM_SUBJECTS = 27
TRIALS_PER_SUBJECT = 3
NUM_CHANNELS = 10
NUM_GLOVE_CHANNELS = 22

SAMPLING_RATE = 100  # Hz (Ninapro DB1)

# NinaPro DB1 label 0 is always "rest"; 1..N are the exercise's gestures.
REST_LABEL = 0

# -------------------------------------------------
# Windowing
# -------------------------------------------------

WINDOW_SIZE_MS = 200
WINDOW_OVERLAP = 0.50

# -------------------------------------------------
# Signal Filtering
# -------------------------------------------------

LOWCUT = 20
HIGHCUT = 450
NOTCH_FREQ = 50

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------

# Used by any future train/test split or model-training code so results
# are reproducible run to run. Do not change this once results have been
# reported without noting it explicitly (e.g. in PROJECT_STATUS.md).
RANDOM_SEED = 42

# -------------------------------------------------
# Train / Test Split
# -------------------------------------------------

# NinaPro DB1 (Exercise A) has 10 repetitions per gesture (rerepetition
# values 1-10; 0 marks rest between them). These defaults hold out
# repetitions {2, 5, 7} for testing and train on the remainder, which is
# the split most commonly used in published NinaPro benchmarks. They are
# only defaults - src/data/splitter.py accepts overrides, and this choice
# should be stated explicitly in any paper/report that relies on it.
DEFAULT_TEST_REPETITIONS = [2, 5, 7]

# -------------------------------------------------
# Create output folders automatically
# -------------------------------------------------

for folder in [
    OUTPUT_DIR,
    VALIDATION_DIR,
    EDA_DIR,
    PREPROCESS_DIR,
    FEATURE_DIR,
    MODEL_DIR,
    REPORT_DIR,
    EXPERIMENTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
