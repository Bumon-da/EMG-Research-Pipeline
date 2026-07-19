"""
Global configuration for the EMG Research Pipeline.
"""

from pathlib import Path

# -------------------------------------------------
# Project Paths
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw" / "DB1_Extracted"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "output"

VALIDATION_DIR = OUTPUT_DIR / "validation"
EDA_DIR = OUTPUT_DIR / "eda"
PREPROCESS_DIR = OUTPUT_DIR / "preprocessing"
FEATURE_DIR = OUTPUT_DIR / "features"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

# -------------------------------------------------
# Dataset
# -------------------------------------------------

NUM_SUBJECTS = 27
TRIALS_PER_SUBJECT = 3
NUM_CHANNELS = 10

SAMPLING_RATE = 100  # Hz (Ninapro DB1)

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
]:
    folder.mkdir(parents=True, exist_ok=True)