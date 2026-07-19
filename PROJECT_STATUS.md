# EMG Research Pipeline

**Project:** EMG Research Pipeline for Hand Rehabilitation Research

**Author:** Priyangshu Protim Gogoi

**Language:** Python 3.13+

**Status:** Active Development

---

# Current Version

v0.2.0

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

---

# Current Architecture

```
main.py
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
    ├── DatasetValidator
    │
    ├── EDAAnalyzer
    │
    ├── Preprocessing (planned)
    │
    ├── Feature Extraction (planned)
    │
    ├── Models (planned)
    │
    └── Evaluation (planned)
```

---

# Project Structure

```
EMG-Research-Pipeline/

config/

data/
    raw/
    processed/
    external/

output/

src/

    data/
        datamodels.py
        loader.py

    managers/
        dataset_manager.py
        results_manager.py
        experiment_manager.py

    eda/
        analyzer.py
        statistics.py
        validator.py

    preprocessing/

    features/

    models/

    evaluation/

tests/

main.py
```

---

# Completed Modules

## Core Infrastructure

- [x] Logging
- [x] Settings
- [x] Data Models
- [x] Dataset Loader
- [x] Dataset Manager
- [x] Results Manager
- [x] Experiment Manager

---

## Dataset

- [x] Automatic loading
- [x] Subject parsing
- [x] Trial parsing
- [x] MAT file reading

---

## Validation

Implemented:

- Missing values
- NaN values
- Infinite values
- Empty channels
- Constant channels

Outputs:

output/validation/

- validation_summary.csv
- validation_details.csv

---

## Exploratory Data Analysis

Implemented

- Dataset summary
- Descriptive statistics
- CSV export

Outputs

output/eda/

- dataset_summary.csv
- per-trial statistics

---

# Current Dataset

Database

Ninapro DB1

Current Statistics

Subjects: 27

Trials: 81

Channels: 10

Total Samples:

12,553,611

Sampling Rate:

100 Hz

---

# Next Milestones

## v0.3.0

Dataset Validation (Advanced)

- Duplicate detection
- Signal ranges
- RMS checks
- Trial duration verification
- Integrity report

---

## v0.4.0

Visualization

- Raw signals
- Histograms
- Boxplots
- Heatmaps
- Subject comparison
- Gesture distribution

---

## v0.5.0

Signal Preprocessing

- Bandpass filter
- Notch filter
- Rectification
- Normalization
- Windowing
- Segmentation

---

## v0.6.0

Feature Extraction

Time Domain

- RMS
- MAV
- WL
- SSC
- ZC
- IEMG

Frequency Domain

- MDF
- MNF
- PSD

Wavelet Features

---

## v0.7.0

Machine Learning

Traditional

- Random Forest
- SVM
- XGBoost

Deep Learning

- CNN
- LSTM

Evaluation

- Accuracy
- Precision
- Recall
- F1
- ROC
- Confusion Matrix

---

## v1.0.0

Complete Research Pipeline

- Automated reports
- HTML documentation
- Experiment tracking
- EMS integration
- Dissertation-ready outputs

---

# Git Milestones

Current Version

v0.2.0

Latest Commit

feat: add dataset validation module and integrate pipeline

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

---

# Current Focus

Working on:

Dataset Validation (Advanced)

Next:

Visualization

Status:

Pipeline Stable
Ready for v0.3.0