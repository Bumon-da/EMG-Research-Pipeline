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

# PROCESSED_DATA_DIR and PREPROCESS_DIR (below) are currently unused -
# actual pipeline output (validation/eda/preprocessing) is written under
# a per-run folder via ResultsManager(output_root=experiment.path), not
# to these global paths. Kept for now; removing them is an unrelated
# cleanup.
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

# These are tuned for raw, AC-coupled sEMG at a realistic sampling rate -
# they are NOT valid for SAMPLING_RATE=100 (Nyquist=50Hz: HIGHCUT exceeds
# it, NOTCH_FREQ sits exactly on it), and NinaPro DB1's `emg` field is
# already a rectified, non-negative sensor envelope, not raw sEMG (see
# src/data/validator.py). src/preprocessing/filtering.py implements a
# bandpass/notch filter using these values but is not invoked against
# DB1 - it's reserved for a future raw-sEMG source (e.g. real hardware).
LOWCUT = 20
HIGHCUT = 450
NOTCH_FREQ = 50

# -------------------------------------------------
# Feature Extraction (v0.6.0)
# -------------------------------------------------

# Whether the train/test split used for feature extraction excludes rest
# (label 0) samples. This is a correctness fix, not just a volume/
# imbalance optimization: NinaPro DB1's rest periods never fall in
# DEFAULT_TEST_REPETITIONS, so with exclude_rest=False the test split
# would contain zero rest windows while train is dominated by them - the
# largest class would never appear in test. See src/features/report.py.
EXCLUDE_REST = True

# NinaPro DB1's gesture labels reset to 1 at the start of every exercise
# file (confirmed: E1 restimulus ranges 0-12, E2 0-17, E3 0-23 - see
# README.md). A per-subject feature file spans all three exercises, so a
# raw `Label` column alone would silently collapse 52 distinct gestures
# into ~23 classes if two exercises' labels were ever compared directly.
# EXERCISE_NUM_GESTURES records each exercise's gesture count (excluding
# rest); src/features/extractor.py derives cumulative offsets from it to
# build a GlobalLabel column (1-52, unique across all three exercises).
EXERCISE_NUM_GESTURES = {1: 12, 2: 17, 3: 23}

# Deadzone (in z-score units) below which a crossing isn't counted, for
# both MCR (mean-crossing rate) and SSC (slope sign changes). 0.0 (no
# deadzone) is the default - not a validated choice, just the simplest
# one that avoids introducing an unjustified magic threshold; revisit if
# quantization noise turns out to inflate either count in practice.
MCR_THRESHOLD = 0.0
SSC_THRESHOLD = 0.0

# Interior histogram bin edges (in z-score units, i.e. applied to the
# per-subject-normalized signal), giving 5 bins total (4 interior edges +
# unbounded outer bins). NOT Atzori's HIST20 - at a 20-sample window,
# 20 bins leaves a measured 1.82 non-empty bins/channel-window (91% of
# columns structurally zero). These edges are also NOT derived per-window
# (that would make the feature incomparable across rows) - they are fixed
# constants chosen to roughly bisect the z-scored envelope's typical
# range (most mass below the mean, a long right tail above it). See
# src/features/report.py for the literature-comparability caveat.
HIST_BIN_EDGES = [-0.5, 0.0, 0.5, 1.5]

# Atzori et al. 2014 used db7 at 5 decomposition levels on much longer
# windows. Neither is possible on this dataset's 20-sample window:
# pywt.dwt_max_level(20, db7) == 0 (db7's 14-tap filter cannot decompose
# it at all). db2 (4-tap) is the shortest common wavelet that still
# permits a useful decomposition (dwt_max_level(20, db2) == 2). This is a
# deliberate deviation from the paper - see src/features/report.py.
WAVELET = "db2"
WAVELET_LEVEL = 2

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
