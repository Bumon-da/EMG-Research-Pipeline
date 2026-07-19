from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "DB1_Extracted"

OUTPUT = PROJECT_ROOT / "output"

SAMPLING_RATE = 100

NUM_CHANNELS = 10

WINDOW_SIZE_MS = 200

WINDOW_OVERLAP = 0.5

LOWCUT = 20

HIGHCUT = 450

NOTCH_FREQ = 50

NUM_SUBJECTS = 27

TRIALS_PER_SUBJECT = 3