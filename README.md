# EMG Research Pipeline

A modular, research-grade Python pipeline for surface EMG (sEMG) based hand-gesture
intent recognition, developed in support of a dissertation/research paper on
machine-learning-based EMG intent recognition for hand rehabilitation assistance.

---

## Dataset

This pipeline is built around **NinaPro DB1** (Atzori et al.), a public sEMG
benchmark: 27 intact-limb subjects, 10-channel EMG sampled at 100 Hz, performing
scripted hand/wrist/finger gestures across three exercises (A/B/C), each with 10
repetitions per gesture.

Each `.mat` file contains:

| Field | Meaning |
|---|---|
| `emg` | (samples, 10) EMG signal - see the envelope note below before assuming this is raw sEMG |
| `glove` | (samples, 22) Cyberglove joint-angle signal |
| `stimulus` / `repetition` | raw gesture / repetition labels (reaction-time delayed) |
| `restimulus` / `rerepetition` | **refined** gesture / repetition labels, corrected for onset delay |

**`emg` is not raw broadband sEMG.** Values are non-negative and quantized
(~0.0024 steps), consistent with the Otto Bock 13E200 sensor's onboard
rectified/enveloped output rather than an AC-coupled waveform - see
`src/data/validator.py`. This is why no bandpass/notch filter is applied
in this pipeline (v0.5.0's filter values are also mathematically invalid
at this dataset's 100 Hz sampling rate regardless); see
`src/preprocessing/filtering.py` and `PROJECT_STATUS.md`'s Key Findings.

The loader reads all of these. Downstream code should default to
`restimulus`/`rerepetition` via `Trial.labels` / `Trial.reps` rather than the raw
fields — the raw `stimulus` label is stamped from a fixed reaction-time offset and
does not line up with true muscle-activation onset, which is why essentially all
published NinaPro results are trained on the refined labels instead.

**Gesture labels are exercise-relative, not global.** Each exercise file
(`_E1`/`_E2`/`_E3`) numbers its gestures starting at 0 (rest) independently -
confirmed by inspection: `restimulus` ranges over 0-12 in E1, 0-17 in E2, and
0-23 in E3 for the same subject. Label `5` in E1 is not the same gesture as
label `5` in E2. Any aggregation across exercises (class-balance reports,
plots, per-class metrics) must key on `(exercise, label)`, never on label
alone - `src/data/validator.py` and `src/eda/visualization.py` do this
correctly; keep it in mind if you add new analysis code.

Because NinaPro DB1 is healthy, intact-limb subjects performing lab-scripted
gestures, it is used here as a benchmarking/feasibility dataset for the
recognition pipeline itself, not as rehabilitation-patient data — worth stating
explicitly in any paper that also frames the project around rehabilitation
assistance.

Cite the dataset as: Atzori, M. et al. (2014), *"Electromyography data for
non-invasive naturally-controlled robotic hand prostheses."* Scientific Data.

### Pointing the pipeline at the dataset

`config/settings.py` resolves the raw data directory in this order:

1. `EMG_RAW_DATA_DIR` environment variable, if set.
2. `data/raw/DB1_Extracted` (the default location).

To point explicitly at another location:

```powershell
# PowerShell
$env:EMG_RAW_DATA_DIR = "C:\path\to\DB1_Extracted"
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

Each run creates a timestamped folder under `output/experiments/` containing
that run's validation report, EDA output, and figures, plus a `manifest.json`
recording the config used and basic dataset stats — so results from different
runs never overwrite each other.

## Tests

```bash
pytest
```

Tests run against synthetic data and do not require the real dataset.

## Dashboard

An interactive Streamlit dashboard for browsing the dataset and pipeline
results:

```bash
streamlit run app.py
```

This opens in your browser (default `http://localhost:8501`). Pages:

- **Home** — dataset overview (subject/trial counts on disk) and a
  "Run Full Pipeline Now" button that runs load → validate → EDA → split →
  preprocess into a new experiment folder without leaving the browser.
- **Raw Signal Browser** — pick a subject and trial, choose channels and a
  time window, and view the raw EMG (and glove, if present) signal
  interactively, with the gesture-label track lined up underneath. Loads
  exactly one `.mat` file on demand — never the whole dataset — so this
  works even before running the full pipeline.
- **Validation** — integrity-check results for a chosen run: summary
  metrics, a filterable flagged-trials table, and the full text report.
- **Gesture Distribution** — sample counts per gesture, one chart per
  exercise (never merged across exercises — see the labeling note above).
- **Cross-Subject Comparison** — mean signal amplitude (RMS) and
  data-quality issue counts across all subjects in a run, built from
  `validation_details.csv` (no raw data reload).
- **Preprocessing** — per-subject normalization statistics, window
  counts by exercise/split, and drop-reason breakdown (transition vs.
  train/test boundary) for a chosen run.

The Validation, Gesture Distribution, Cross-Subject Comparison, and Preprocessing pages
read the CSV/JSON output of a pipeline run (`output/experiments/<run>/`),
so run the pipeline at least once — either `python main.py` or the Home
page's button — before expecting data there. The Raw Signal Browser reads
`.mat` files directly and doesn't need a prior run.

---

## Project Status

See `PROJECT_STATUS.md` for the current architecture, completed modules, and
roadmap.

## Development Philosophy

- Modular architecture, single-responsibility modules
- Type hints throughout
- Logging instead of print for pipeline-internal messages
- All outputs go through `ResultsManager`
- Full-file updates, no partial patches, during development
- Reproducibility: fixed random seed (`config.settings.RANDOM_SEED`), documented
  train/test split strategy, per-run output isolation
