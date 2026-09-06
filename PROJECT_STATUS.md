# EMG Research Pipeline

**Project:** EMG Research Pipeline for Hand Rehabilitation Research

**Author:** Priyangshu Protim Gogoi

**Language:** Python 3.13+

**Status:** Active Development

---

# Current Version

v0.4.0

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
    ├── Preprocessing (planned)
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

pages/
    1_Raw_Signal_Browser.py
    2_Validation.py
    3_Gesture_Distribution.py
    4_Cross_Subject_Comparison.py

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

Implemented (`src/data/splitter.py`), not yet consumed by a training stage:

- `repetition_split`: subject-dependent, split by repetition (default test reps [2, 5, 7])
- `subject_split`: subject-independent, leave-N-subjects-out
- Returns per-trial boolean masks (not per-sample objects) for memory/performance reasons at this dataset's scale (~12.5M samples)
- `RANDOM_SEED` centralized in `config/settings.py`

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

Architecture note: `src/pipeline.py` now holds the single
load-validate-EDA sequence that both `main.py` and the dashboard's "Run
Pipeline Now" button call, so the two entry points can't drift apart.
Aggregate dashboard views (Validation, Gesture Distribution,
Cross-Subject Comparison) read the CSV/JSON a run already produced rather
than reloading the dataset; only the Raw Signal Browser touches `.mat`
files directly, and only the one file being viewed.

---

# Key Findings From This Pass (worth stating explicitly in any paper)

1. **Refined labels matter.** `restimulus`/`rerepetition` correct a reaction-time delay present in raw `stimulus`/`repetition`. The loader now reads both; `Trial.labels`/`Trial.reps` default to the refined fields.
2. **`glove` (22-channel joint-angle) data exists in every file and is now captured**, even though nothing consumes it yet - available for a future auxiliary/validation signal.
3. **NinaPro DB1's `emg` field is not raw broadband sEMG.** Values are non-negative, quantized (~0.0024 steps), consistent with the Otto Bock 13E200 sensor's onboard rectified/enveloped output rather than an AC-coupled waveform. The bandpass/notch filter settings in `config/settings.py` (20-450 Hz / 50 Hz notch) target raw sEMG and should be reconsidered for this signal when preprocessing is implemented (v0.5.0).
4. **Gesture labels reset per exercise file.** Label 5 in `_E1` is not the same gesture as label 5 in `_E2`/`_E3`. All aggregation in this codebase now keys on `(exercise, label)` - this was a real bug caught and fixed during this pass (class distribution was initially merging labels across exercises).
5. **`requirements.txt` was UTF-16 encoded** (likely `pip freeze` from PowerShell) - re-saved as UTF-8, since this can break `pip install -r` on some setups. (Caught and fixed twice during this project's development - worth double-checking with `file requirements.txt` after any edit to this file specifically, since a plain-text non-Python file's encoding won't surface as an import error the way a corrupted `.py` file would.)

---

# Next Milestones

## v0.5.0

Signal Preprocessing

- Decide filtering approach appropriate for DB1's already-rectified signal (see Key Finding #3)
- Rectification / normalization (revisit given #3)
- Windowing (`WINDOW_SIZE_MS` / `WINDOW_OVERLAP` already configured, unused so far)
- Segmentation, applied per the chosen split strategy's train/test masks

---

## v0.6.0

Feature Extraction

Time Domain: RMS, MAV, WL, SSC, ZC, IEMG

Frequency Domain: MDF, MNF, PSD

Wavelet Features

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

v0.4.0

Latest Change

feat: interactive Streamlit dashboard (raw signal browser, validation, gesture distribution, cross-subject comparison), pipeline logic consolidated into src/pipeline.py, per-trial mean RMS, on-demand single-trial loader

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

Deciding the preprocessing/filtering approach for v0.5.0, informed by Key Finding #3 above (DB1's `emg` is a rectified sensor envelope, not raw sEMG)

Next:

Signal Preprocessing (v0.5.0)

Status:

Pipeline Stable. Advanced validation, visualization, interactive dashboard, and split strategy in place. Ready for v0.5.0.
