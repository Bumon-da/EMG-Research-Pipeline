# EMG Research Pipeline

**Project:** EMG Research Pipeline for Hand Rehabilitation Research

**Author:** Priyangshu Protim Gogoi

**Language:** Python 3.14+ (developed/tested on 3.14.4)

**Status:** Active Development

---

# Current Version

v0.5.1

---

# Project Goal

Develop a modular, research-grade EMG processing framework capable of:

- Dataset validation
- Exploratory Data Analysis (EDA)
- Signal preprocessing
- Feature extraction
- Machine Learning
- Gesture prediction
- EMS integration
- Automated report generation

The framework is intended to support dissertation work while remaining reusable for future research.

---

# Development Philosophy

This project follows professional software engineering principles.

Guidelines:

- Modular architecture
- Single Responsibility Principle (SRP)
- Type hints
- Logging
- Clean folder structure
- Full-file updates (no patch snippets)
- Git versioning
- Research-grade documentation
- Reproducibility: fixed random seed, documented train/test split strategy, per-run output isolation

---

# Current Architecture

```
main.py / app.py (Streamlit dashboard)
    │
    ▼
src.pipeline.run_pipeline  <- single source of truth for a full run
    │
    ▼
ExperimentManager (per-run output isolation)
    │
    ▼
DatasetManager
    │
    ▼
DatasetLoader
    │
    ▼
Subjects
    │
    ├── DatasetValidator (basic + advanced integrity checks)
    │
    ├── EDAAnalyzer + Visualizer
    │
    ├── Splitter (repetition_split / subject_split)
    │
    ├── Preprocessing (rectification guard, per-subject normalization,
    │                   windowing/segmentation - v0.5.0)
    │
    ├── Feature Extraction (planned)
    │
    ├── Models (planned)
    │
    └── Evaluation (planned)

Streamlit dashboard (app.py + pages/) reads pipeline output (CSV/JSON)
for aggregate views, and loads single .mat files on demand for the raw
signal browser - it does not reload the whole dataset per interaction.
```

---

# Project Structure

```
EMG-Research-Pipeline/

config/
    settings.py
    logging_config.py

data/
    raw/
    processed/
    external/

output/
    experiments/<run_id>/   <- all outputs for one run live here

src/

    data/
        datamodels.py
        loader.py
        validator.py
        splitter.py

    managers/
        dataset_manager.py
        results_manager.py
        experiment_manager.py

    eda/
        analyzer.py
        statistics.py
        visualization.py
        report.py

    preprocessing/
        normalization.py
        filtering.py
        segmentation.py
        preprocessor.py
        report.py

    features/

    models/

    evaluation/

    dashboard/
        data_access.py
        state.py

    pipeline.py

tests/
    conftest.py
    test_datamodels.py
    test_statistics.py
    test_splitter.py
    test_validator.py
    test_dashboard_data_access.py
    test_normalization.py
    test_filtering.py
    test_segmentation.py
    test_preprocessor.py
    test_results_manager.py

pages/
    1_Raw_Signal_Browser.py
    2_Validation.py
    3_Gesture_Distribution.py
    4_Cross_Subject_Comparison.py
    5_Preprocessing.py

app.py       <- Streamlit dashboard entry point (streamlit run app.py)
main.py      <- CLI entry point
```

---

# Completed Modules

## Core Infrastructure

- [x] Logging
- [x] Settings (with EMG_RAW_DATA_DIR override, RANDOM_SEED, split config)
- [x] Data Models (now carry restimulus/rerepetition/glove)
- [x] Dataset Loader (loads refined labels + glove; enforces array-length consistency)
- [x] Dataset Manager
- [x] Results Manager (CSV/text/JSON/figures; supports per-run output roots)
- [x] Experiment Manager (per-run isolated output folder + manifest.json)

---

## Dataset

- [x] Automatic loading
- [x] Subject parsing
- [x] Trial parsing
- [x] MAT file reading (emg, glove, stimulus/restimulus, repetition/rerepetition, exercise)

---

## Validation (v0.3.0 - Advanced)

Implemented:

- Missing / NaN / infinite values
- Negative values (data-integrity flag specific to DB1's non-negative sensor output)
- Empty channels
- Constant channels (tolerance-based, not exact float equality)
- Duplicate-trial detection (content hash)
- Saturation / clipping detection (count + fraction based, tuned to avoid false positives on small samples)
- Low-activity channel detection (possible poor electrode contact)
- Trial duration outlier detection (per exercise)
- Gesture class distribution + imbalance ratio, correctly keyed by (exercise, label)

Outputs (per run, under `output/experiments/<run_id>/validation/`):

- validation_details.csv
- validation_summary.csv
- flagged_trials.csv
- class_distribution.csv
- integrity_report.txt

---

## Exploratory Data Analysis (v0.4.0 - Visualization)

Implemented:

- Dataset summary
- Descriptive statistics per trial
- Raw signal plots, channel histograms/boxplots, channel-correlation heatmap (sampled trials)
- Subject comparison (mean RMS by subject)
- Gesture class distribution, faceted by exercise
- Short text EDA report

Outputs: `output/experiments/<run_id>/eda/`

---

## Train / Test Split Strategy

Implemented (`src/data/splitter.py`); consumed by Signal Preprocessing (v0.5.0, below) for leakage-free normalization and split-aware windowing:

- `repetition_split`: subject-dependent, split by repetition (default test reps [2, 5, 7])
- `subject_split`: subject-independent, leave-N-subjects-out
- Returns per-trial boolean masks (not per-sample objects) for memory/performance reasons at this dataset's scale (~12.5M samples)
- `RANDOM_SEED` centralized in `config/settings.py`

---

## Signal Preprocessing (v0.5.0)

Implemented (`src/preprocessing/`):

- **Filtering decision (resolves Key Finding #3):** no bandpass/notch
  filter is applied to NinaPro DB1. Its `emg` field is already a
  rectified, non-negative sensor envelope, not raw sEMG, and the
  configured filter values (`LOWCUT=20`, `HIGHCUT=450`, `NOTCH_FREQ=50`)
  are mathematically invalid for `SAMPLING_RATE=100` anyway (Nyquist =
  50Hz: `HIGHCUT` exceeds it, `NOTCH_FREQ` sits exactly on it).
  `src/preprocessing/filtering.py`'s `bandpass_notch_filter()` implements
  a real, tested Butterworth bandpass + IIR notch (and validates its own
  arguments against Nyquist) for a future raw-sEMG source - it is not
  invoked anywhere in the DB1 path.
- **Rectification guard** (`Normalizer.clean`): clips stray negative
  samples to 0 and neutralizes non-finite samples, so a single corrupted
  trial can't silently poison a subject's normalization statistics. On
  the real dataset today this clips 0 samples (validation already shows
  0 negative/NaN/Inf values) - it exists for robustness, not because
  today's data needs it.
- **Per-subject, per-channel z-score normalization** (`Normalizer`),
  computed from each subject's TRAIN-split samples only (via
  `src/data/splitter.py` - its first real consumer since being built in
  v0.4.0) and applied to both that subject's train and test rows, so
  test-split values never influence the statistics used to scale them.
  Falls back to a subject's full data if it has zero train samples (a
  `subject_split`-only edge case), flagged via `UsedTestFallback`.
- **Windowing/segmentation** (`src/preprocessing/segmentation.py`):
  `WINDOW_SIZE_MS`/`WINDOW_OVERLAP` (200ms/50% -> 20-sample windows,
  10-sample stride) are now consumed. A candidate window is kept only if
  its label is uniform (drop label-transition windows) and it falls
  entirely inside one side of the train/test split (drop windows
  straddling the boundary - the rule `src/data/splitter.py`'s own
  docstring anticipated). Windows carry index metadata only
  (`start`/`end`/`label`/`split`), never a copy of the signal -
  `trial.emg[start:end]` is a zero-copy view for feature extraction to
  use later.

Outputs (per run, under `output/experiments/<run_id>/preprocessing/`):

- normalization_stats.csv, rectification_report.csv, window_tally.csv,
  window_drop_summary.csv, preprocessing_summary.json,
  preprocessing_report.txt

On the real 27-subject/81-trial dataset (`repetition_split`, default test
repetitions `[2, 5, 7]`): 1,201,946 windows kept (1,051,484 train /
150,462 test), 53,298 dropped for a label transition, 0 dropped for a
split boundary (the two are logically independent checks - see
`tests/test_segmentation.py` - but happen to coincide on this dataset
since every repetition boundary is also a label transition), 0 negative
or non-finite samples clipped.

Not included in this milestone (deliberately deferred): a
`pages/5_Preprocessing.py` dashboard page. This project's own history
shows the Validation page shipped a full version after the validation
logic itself (v0.3.0 logic -> v0.4.0 page); the read-side plumbing
(`ExperimentRun.preprocessing_dir` + 4 loader functions in
`src/dashboard/data_access.py`) is in place so a page is a small
follow-up whenever it's wanted.

---

## Interactive Dashboard (v0.4.0)

Implemented (`app.py` + `pages/`, run with `streamlit run app.py`):

- **Home** - dataset overview from disk (no full load required) + a
  "Run Full Pipeline Now" button that runs `src.pipeline.run_pipeline`
  from the browser
- **Raw Signal Browser** - subject/trial selection, on-demand single-trial
  load, channel + time-window controls, interactive signal plot with the
  gesture-label track synced underneath, optional glove overlay,
  descriptive stats table
- **Validation** - run picker, summary metrics, filterable flagged-trials
  table, full integrity report
- **Gesture Distribution** - run picker, one interactive chart per
  exercise (never merged - see Key Finding #4)
- **Cross-Subject Comparison** - mean RMS and data-quality issue counts
  across all subjects, built from `validation_details.csv` (no raw data
  reload)
- **Preprocessing** (added v0.5.1) - run picker, headline metrics from
  `preprocessing_summary.json`, per-subject normalization statistics
  table, window counts by exercise/split, drop-reason breakdown
  (transition vs. train/test boundary), rectification report, full text
  report - the page deferred at the end of v0.5.0, once the read-side
  plumbing (`ExperimentRun.preprocessing_dir` + loaders in
  `src/dashboard/data_access.py`) already existed

Architecture note: `src/pipeline.py` now holds the single
load-validate-EDA-split-preprocess sequence that both `main.py` and the
dashboard's "Run Pipeline Now" button call, so the two entry points can't
drift apart. Aggregate dashboard views (Validation, Gesture Distribution,
Cross-Subject Comparison, Preprocessing) read the CSV/JSON a run already
produced rather than reloading the dataset; only the Raw Signal Browser
touches `.mat` files directly, and only the one file being viewed.

---

## v0.5.1 Housekeeping

Small cleanup pass ahead of v0.6.0 Feature Extraction - no new pipeline
stage, but two changes remove friction for it:

- **`SplitResult.get()`** (`src/data/splitter.py`) now does an O(1) dict
  lookup instead of a linear scan over every `TrialSplit`, via a lazily
  built index that rebuilds itself if more trials are appended after the
  first `get()` call.
- **`ResultsManager.save_parquet()`** added, mirroring `save_csv`'s
  signature/logging - v0.6.0's feature matrices will be too large for the
  CSV convention used everywhere else in this pipeline. `pyarrow` pinned
  explicitly in `requirements.txt` (was previously present only as a
  transitive pandas 3.x dependency).
- Removed `pipeline diagrams/evaluation_metrics.xml` (a byte-identical,
  misnamed duplicate of `ems_feedback.xml`) and the stale
  `setup_project.py` scaffolder (predated `src/managers/`,
  `src/dashboard/`, `pages/`, `src/pipeline.py`).
- Corrected inaccuracies: `README.md`'s field table called `emg` "raw
  surface EMG" (contradicts Key Finding #3 below - it's an already-
  rectified sensor envelope); `PROJECT_STATUS.md` said "Python 3.13+"
  when runs are on 3.14.4; the Home page's button description undersold
  what it actually runs.

---

# Key Findings From This Pass (worth stating explicitly in any paper)

1. **Refined labels matter.** `restimulus`/`rerepetition` correct a reaction-time delay present in raw `stimulus`/`repetition`. The loader now reads both; `Trial.labels`/`Trial.reps` default to the refined fields.
2. **`glove` (22-channel joint-angle) data exists in every file and is now captured**, even though nothing consumes it yet - available for a future auxiliary/validation signal.
3. **NinaPro DB1's `emg` field is not raw broadband sEMG.** Values are non-negative, quantized (~0.0024 steps), consistent with the Otto Bock 13E200 sensor's onboard rectified/enveloped output rather than an AC-coupled waveform. The bandpass/notch filter settings in `config/settings.py` (20-450 Hz / 50 Hz notch) target raw sEMG and should be reconsidered for this signal when preprocessing is implemented (v0.5.0). **Resolved in v0.5.0:** the filter is not applied to DB1 (per-subject z-score normalization is used instead); it's also mathematically invalid for `SAMPLING_RATE=100` regardless (`HIGHCUT`/`NOTCH_FREQ` violate/hit the 50Hz Nyquist limit) - see "Signal Preprocessing (v0.5.0)" above and `src/preprocessing/filtering.py`.
4. **Gesture labels reset per exercise file.** Label 5 in `_E1` is not the same gesture as label 5 in `_E2`/`_E3`. All aggregation in this codebase now keys on `(exercise, label)` - this was a real bug caught and fixed during this pass (class distribution was initially merging labels across exercises).
5. **`requirements.txt` was UTF-16 encoded** (likely `pip freeze` from PowerShell) - re-saved as UTF-8, since this can break `pip install -r` on some setups. (Caught and fixed twice during this project's development - worth double-checking with `file requirements.txt` after any edit to this file specifically, since a plain-text non-Python file's encoding won't surface as an import error the way a corrupted `.py` file would.)

---

# Next Milestones

v0.5.0 (Signal Preprocessing) is complete - see "Signal Preprocessing (v0.5.0)" above.

## v0.6.0

Feature Extraction

Time Domain: RMS, MAV, WL, SSC, ZC, IEMG

Frequency Domain: MDF, MNF, PSD

Wavelet Features

Consumes `src/preprocessing/segmentation.py`'s `segment_trial()` /
`Window` (index metadata only - slice `trial.emg[start:end]` per window)
and `src/preprocessing/normalization.py`'s `Normalizer.apply()` with a
run's saved `SubjectNormalizationStats`, computing one feature vector per
window rather than materializing the full window index or a normalized
dataset copy.

---

## v0.7.0

Machine Learning

Traditional: Random Forest, SVM, XGBoost

Deep Learning: CNN, LSTM

Evaluation: Accuracy, Precision, Recall, F1, ROC, Confusion Matrix (per-exercise, since label sets differ)

---

## v1.0.0

Complete Research Pipeline

- Automated reports
- HTML documentation
- Experiment tracking (foundation now in place via ExperimentManager)
- EMS integration
- Dissertation-ready outputs

---

# Git Milestones

Current Version

v0.5.1

Latest Change

chore: v0.5.1 housekeeping ahead of v0.6.0 - pages/5_Preprocessing.py dashboard page (the deferred v0.5.0 follow-up), SplitResult.get() now O(1) (dict index instead of a linear scan), ResultsManager.save_parquet() added ahead of v0.6.0's feature matrices, removed the byte-identical duplicate diagram file and the stale setup_project.py scaffolder, corrected README/PROJECT_STATUS inaccuracies (emg raw-sEMG claim, Python version, Home button's actual pipeline steps)

---

# Development Rules

Always:

✔ Update PROJECT_STATUS.md after every milestone

✔ Commit after each completed module

✔ Push to GitHub

✔ Maintain modular architecture

✔ Keep code fully typed

✔ Use ResultsManager for saving outputs

✔ Use logging instead of print where appropriate

✔ Provide complete updated files during development

✔ Run `pytest` before considering a change done

---

# Current Focus

Working on:

v0.6.0, Feature Extraction (Atzori et al. 2014 DB1 baseline: RMS, MAV, WL, SSC, ZC/MCR, IEMG, HIST, mDWT), consuming the windows and normalization stats produced by v0.5.0. Frequency-domain features (MDF/MNF/PSD) are deliberately dropped - see the Key Findings this milestone will add.

Next:

Feature Extraction (v0.6.0)

Status:

Pipeline Stable. Advanced validation, visualization, interactive dashboard, split strategy, and signal preprocessing (rectification, per-subject normalization, windowing/segmentation) all in place, including the Preprocessing dashboard page deferred from v0.5.0. Ready for v0.6.0.
