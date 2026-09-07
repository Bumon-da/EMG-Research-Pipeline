import json

import numpy as np
import pandas as pd

from src.data.datamodels import Subject
from src.data.splitter import repetition_split, subject_split
from src.features.extractor import FeatureExtractor, exercise_label_offsets
from src.managers.results_manager import ResultsManager
from src.preprocessing.normalization import SubjectNormalizationStats
from tests.conftest import make_subject, make_trial


def make_extractor(tmp_path):
    return FeatureExtractor(results=ResultsManager(output_root=tmp_path))


def make_non_negative_emg(n_samples, n_channels, seed=0):
    """DB1-like: non-negative, so a naive (non-demeaned) crossing test on
    the raw signal would be structurally 0 - see test_time_domain.py."""
    return np.random.default_rng(seed).random((n_samples, n_channels)) + 0.1


def build_simple_subjects(n_subjects=2, n_samples=200, n_channels=3):
    """
    Labels 1/2 (never 0/rest) over 4 repetitions, so exclude_rest=True
    still leaves every sample eligible and windows split cleanly between
    train/test by repetition.
    """
    subjects = []
    for i in range(n_subjects):
        emg = make_non_negative_emg(n_samples, n_channels, seed=i)
        quarter = n_samples // 4
        labels = np.array([1] * (quarter * 2) + [2] * (quarter * 2))
        reps = np.array([1] * quarter + [2] * quarter + [3] * quarter + [4] * quarter)
        trial = make_trial(
            filename=f"S{i}_A1_E1.mat",
            n_samples=n_samples,
            n_channels=n_channels,
            emg=emg,
            restimulus=labels,
            rerepetition=reps,
            stimulus=labels,
            repetition=reps,
        )
        subjects.append(make_subject(f"S{i}", trials=[trial]))
    return subjects


def test_extract_saves_expected_output_files(tmp_path):
    subjects = build_simple_subjects()
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    make_extractor(tmp_path).extract(subjects, split_result)

    features_dir = tmp_path / "features"
    assert (features_dir / "S0_features.parquet").exists()
    assert (features_dir / "S1_features.parquet").exists()
    assert (features_dir / "normalization_stats.csv").exists()
    assert (features_dir / "feature_manifest.csv").exists()
    assert (features_dir / "feature_summary.json").exists()
    assert (features_dir / "features_report.txt").exists()


def test_extract_feature_manifest_matches_saved_parquet_columns(tmp_path):
    subjects = build_simple_subjects(n_subjects=1)
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    result = make_extractor(tmp_path).extract(subjects, split_result)

    saved = pd.read_parquet(tmp_path / "features" / "S0_features.parquet")
    metadata_columns = {"Subject", "Trial", "Exercise", "Label", "GlobalLabel", "Split", "Start"}
    feature_columns_in_parquet = set(saved.columns) - metadata_columns

    assert feature_columns_in_parquet == set(result.feature_manifest_df["Column"])


def test_extract_produces_one_row_per_kept_window(tmp_path):
    subjects = build_simple_subjects(n_subjects=1, n_samples=200)
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    result = make_extractor(tmp_path).extract(subjects, split_result)

    saved = pd.read_parquet(tmp_path / "features" / "S0_features.parquet")
    assert len(saved) == result.summary["total_windows"]
    assert len(saved) == result.summary["total_windows_train"] + result.summary["total_windows_test"]


def test_extract_excludes_rest_when_split_excludes_rest(tmp_path):
    # A 20-sample window needs 20 CONTIGUOUS samples of the same
    # (label, split) to survive segmentation - so the train/test gesture
    # blocks here are 50 samples each, not just "more than the rest".
    n_samples = 200
    emg = make_non_negative_emg(n_samples, 2)
    # First half rest (label 0), second half a real gesture split 50/50
    # train/test by repetition.
    labels = np.array([0] * 100 + [1] * 100)
    reps = np.array([1] * 100 + [1] * 50 + [2] * 50)
    trial = make_trial(
        filename="S0_A1_E1.mat", n_samples=n_samples, n_channels=2,
        emg=emg, restimulus=labels, rerepetition=reps, stimulus=labels, repetition=reps,
    )
    subjects = [make_subject("S0", trials=[trial])]
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    make_extractor(tmp_path).extract(subjects, split_result)

    saved = pd.read_parquet(tmp_path / "features" / "S0_features.parquet")
    assert not saved.empty
    assert (saved["Label"] != 0).all()
    assert (saved["Split"] == "train").any()
    assert (saved["Split"] == "test").any()


def test_extract_global_label_avoids_collision_across_exercises(tmp_path):
    """
    Pins the highest-severity bug found during design validation: NinaPro
    DB1 restarts gesture numbering at 1 for every exercise file, so the
    same raw Label in two exercises must map to different GlobalLabel
    values, not collide.
    """
    n_samples = 100

    def build_trial(filename, exercise_id, label_value):
        emg = make_non_negative_emg(n_samples, 2, seed=exercise_id)
        labels = np.array([label_value] * n_samples)
        reps = np.array([1] * n_samples)
        trial = make_trial(
            filename=filename, n_samples=n_samples, n_channels=2,
            emg=emg, restimulus=labels, rerepetition=reps, stimulus=labels, repetition=reps,
        )
        trial.exercise = np.array([[exercise_id]])
        return trial

    trial_e1 = build_trial("S0_A1_E1.mat", exercise_id=1, label_value=5)
    trial_e2 = build_trial("S0_A1_E2.mat", exercise_id=2, label_value=5)
    subjects = [make_subject("S0", trials=[trial_e1, trial_e2])]
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    make_extractor(tmp_path).extract(subjects, split_result)

    saved = pd.read_parquet(tmp_path / "features" / "S0_features.parquet")
    e1_global = set(saved[saved["Exercise"] == 1]["GlobalLabel"])
    e2_global = set(saved[saved["Exercise"] == 2]["GlobalLabel"])

    assert e1_global == {5}
    assert e2_global == {5 + exercise_label_offsets()[2]}
    assert e1_global.isdisjoint(e2_global)


def test_extract_mcr_column_is_not_structurally_zero(tmp_path):
    """
    End-to-end pin of the ordering requirement: MCR extracted from the
    per-subject-normalized (not raw) signal should not be all zero, even
    though the trial's raw emg is non-negative.
    """
    subjects = build_simple_subjects(n_subjects=1)
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    make_extractor(tmp_path).extract(subjects, split_result)

    saved = pd.read_parquet(tmp_path / "features" / "S0_features.parquet")
    mcr_columns = [c for c in saved.columns if c.endswith("_MCR")]
    assert mcr_columns
    assert (saved[mcr_columns] != 0).any().any()


def test_extract_feature_dtypes_match_schema(tmp_path):
    subjects = build_simple_subjects(n_subjects=1)
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    make_extractor(tmp_path).extract(subjects, split_result)

    saved = pd.read_parquet(tmp_path / "features" / "S0_features.parquet")
    assert saved["Ch1_RMS"].dtype == np.float32
    assert saved["Ch1_MCR"].dtype == np.int8
    assert saved["Ch1_SSC"].dtype == np.int8
    assert saved["Ch1_HIST_b0"].dtype == np.int8
    assert saved["Ch1_DWT_A2"].dtype == np.float32
    assert saved["Label"].dtype == np.int8
    assert saved["Start"].dtype == np.int32


def test_extract_skips_subject_with_zero_trials(tmp_path):
    subjects = build_simple_subjects(n_subjects=1)
    subjects.append(Subject(subject_id="EMPTY", trials=[]))
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    result = make_extractor(tmp_path).extract(subjects, split_result)

    assert not (tmp_path / "features" / "EMPTY_features.parquet").exists()
    assert result.summary["subjects_with_zero_trials"] == 1


def test_extract_handles_subject_fully_held_out_by_subject_split(tmp_path):
    subjects = build_simple_subjects(n_subjects=2)
    split_result = subject_split(subjects, test_subject_ids=["S0"], exclude_rest=True)

    result = make_extractor(tmp_path).extract(subjects, split_result)

    s0_stats = result.normalization_stats_df[result.normalization_stats_df["Subject"] == "S0"]
    assert not s0_stats.empty
    assert bool(s0_stats.iloc[0]["UsedTestFallback"]) is True
    assert result.summary["subjects_with_train_fallback"] >= 1


def test_extract_uses_precomputed_stats_instead_of_recomputing(tmp_path):
    """
    Proves precomputed_stats is actually consulted (src/pipeline.py passes
    SignalPreprocessor's result.subject_stats), not silently ignored: a
    deliberately-wrong mean (100.0, far from this signal's true ~0.1-1.1
    range) must show up verbatim in the saved output.
    """
    subjects = build_simple_subjects(n_subjects=1)
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    fake_stats = SubjectNormalizationStats(
        subject_id="S0",
        channel_mean=np.array([100.0, 100.0, 100.0]),
        channel_std=np.array([1.0, 1.0, 1.0]),
        train_sample_count=1,
    )

    result = make_extractor(tmp_path).extract(subjects, split_result, precomputed_stats={"S0": fake_stats})

    stats_row = result.normalization_stats_df.iloc[0]
    assert stats_row["Mean"] == 100.0
    assert stats_row["Std"] == 1.0


def test_extract_falls_back_to_computing_stats_for_subject_missing_from_precomputed(tmp_path):
    """A partial precomputed_stats dict (e.g. covering only some subjects)
    must not break or skip the subjects it doesn't cover."""
    subjects = build_simple_subjects(n_subjects=2)
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    result = make_extractor(tmp_path).extract(subjects, split_result, precomputed_stats={})

    assert set(result.normalization_stats_df["Subject"]) == {"S0", "S1"}


def test_extract_global_label_is_sentinel_when_exercise_id_unrecognized(tmp_path):
    """
    Pins the fix for a silent GlobalLabel collision: a trial whose
    exercise_id can't be parsed (Trial.exercise_id returns None) must not
    fall back to offset 0, which would numerically collide with real
    exercise-1 GlobalLabels - see src/features/extractor.py's
    _build_metadata.
    """
    subjects = build_simple_subjects(n_subjects=1)
    subjects[0].trials[0].exercise = np.array([])  # -> Trial.exercise_id is None
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    make_extractor(tmp_path).extract(subjects, split_result)

    saved = pd.read_parquet(tmp_path / "features" / "S0_features.parquet")
    assert not saved.empty
    assert (saved["GlobalLabel"] == -1).all()
    assert (saved["Exercise"] == -1).all()


def test_extract_feature_manifest_reflects_max_channel_count_across_subjects(tmp_path):
    """
    feature_manifest.csv must reflect the widest channel count seen across
    subjects, not whichever subject happened to be processed last (dict/
    list iteration order isn't a channel-count guarantee) - see
    src/features/extractor.py's channel_counts tracking.
    """
    subjects = build_simple_subjects(n_subjects=1, n_channels=2) + build_simple_subjects(
        n_subjects=1, n_channels=4
    )
    subjects[1] = Subject(subject_id="S1", trials=subjects[1].trials)  # avoid duplicate "S0" id
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    result = make_extractor(tmp_path).extract(subjects, split_result)

    assert any(c.startswith("Ch4_") for c in result.feature_manifest_df["Column"])


def test_extract_summary_values_are_native_python_types(tmp_path):
    subjects = build_simple_subjects()
    split_result = repetition_split(subjects, test_repetitions=[2], exclude_rest=True)

    result = make_extractor(tmp_path).extract(subjects, split_result)

    round_tripped = json.loads(json.dumps(result.summary))
    for key, value in result.summary.items():
        assert type(round_tripped[key]) is type(value)
