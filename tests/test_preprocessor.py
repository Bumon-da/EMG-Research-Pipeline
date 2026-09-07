import json

import numpy as np
import pandas as pd

from src.data.datamodels import Subject
from src.data.splitter import repetition_split, subject_split
from src.managers.results_manager import ResultsManager
from src.preprocessing.preprocessor import SignalPreprocessor
from tests.conftest import make_subject, make_trial


def make_preprocessor(tmp_path):
    return SignalPreprocessor(results=ResultsManager(output_root=tmp_path))


def build_simple_subjects(n_subjects=2, n_samples=40, n_channels=3):
    subjects = []
    for i in range(n_subjects):
        rng = np.random.default_rng(i)
        emg = rng.random((n_samples, n_channels)) + 0.1
        labels = np.array([1] * (n_samples // 2) + [2] * (n_samples // 2))
        trial = make_trial(
            filename=f"S{i}_trial.mat",
            n_samples=n_samples,
            n_channels=n_channels,
            emg=emg,
            restimulus=labels,
            rerepetition=labels,
        )
        subjects.append(make_subject(f"S{i}", trials=[trial]))
    return subjects


def test_preprocess_saves_expected_output_files(tmp_path):
    subjects = build_simple_subjects()
    split_result = repetition_split(subjects, test_repetitions=[2])

    make_preprocessor(tmp_path).preprocess(subjects, split_result)

    preprocessing_dir = tmp_path / "preprocessing"
    assert (preprocessing_dir / "normalization_stats.csv").exists()
    assert (preprocessing_dir / "rectification_report.csv").exists()
    assert (preprocessing_dir / "window_tally.csv").exists()
    assert (preprocessing_dir / "window_drop_summary.csv").exists()
    assert (preprocessing_dir / "preprocessing_summary.json").exists()
    assert (preprocessing_dir / "preprocessing_report.txt").exists()


def test_preprocess_returns_dataframes_matching_saved_csvs(tmp_path):
    subjects = build_simple_subjects()
    split_result = repetition_split(subjects, test_repetitions=[2])

    result = make_preprocessor(tmp_path).preprocess(subjects, split_result)

    saved = pd.read_csv(tmp_path / "preprocessing" / "normalization_stats.csv")
    assert len(saved) == len(result.normalization_stats_df)
    assert list(saved.columns) == list(result.normalization_stats_df.columns)


def test_preprocess_normalization_stats_has_one_row_per_subject_channel(tmp_path):
    subjects = build_simple_subjects(n_subjects=3, n_channels=4)
    split_result = repetition_split(subjects, test_repetitions=[2])

    result = make_preprocessor(tmp_path).preprocess(subjects, split_result)

    assert len(result.normalization_stats_df) == 3 * 4


def test_preprocess_handles_subject_fully_held_out_by_subject_split(tmp_path):
    subjects = build_simple_subjects(n_subjects=3)
    split_result = subject_split(subjects, test_subject_ids=["S0"])

    result = make_preprocessor(tmp_path).preprocess(subjects, split_result)

    s0_rows = result.normalization_stats_df[result.normalization_stats_df["Subject"] == "S0"]
    assert not s0_rows.empty
    assert bool(s0_rows.iloc[0]["UsedTestFallback"]) is True
    assert result.summary["subjects_with_train_fallback"] >= 1


def test_preprocess_skips_subject_with_zero_trials_and_logs_it(tmp_path):
    subjects = build_simple_subjects(n_subjects=1)
    subjects.append(Subject(subject_id="EMPTY", trials=[]))
    split_result = repetition_split(subjects, test_repetitions=[2])

    result = make_preprocessor(tmp_path).preprocess(subjects, split_result)

    assert "EMPTY" not in set(result.normalization_stats_df["Subject"])
    assert result.summary["subjects_with_zero_trials"] == 1


def test_preprocess_rectification_report_counts_negative_samples(tmp_path):
    n_samples = 40
    emg = np.random.default_rng(0).random((n_samples, 2)) + 0.1
    emg[0, 0] = -5.0
    labels = np.array([1] * (n_samples // 2) + [2] * (n_samples // 2))
    trial = make_trial(
        filename="t.mat", n_samples=n_samples, n_channels=2, emg=emg, restimulus=labels, rerepetition=labels
    )
    subjects = [make_subject("S0", trials=[trial])]
    split_result = repetition_split(subjects, test_repetitions=[2])

    result = make_preprocessor(tmp_path).preprocess(subjects, split_result)

    row = result.rectification_df.iloc[0]
    assert row["NegativeSamplesClipped"] == 1


def test_preprocess_window_tally_sums_to_manual_count(tmp_path):
    subjects = build_simple_subjects(n_subjects=1, n_samples=40)
    split_result = repetition_split(subjects, test_repetitions=[2])

    result = make_preprocessor(tmp_path).preprocess(subjects, split_result)

    manual_total = (
        result.window_drop_summary_df["KeptTrain"].sum() + result.window_drop_summary_df["KeptTest"].sum()
    )
    assert result.window_tally_df["WindowCount"].sum() == manual_total


def test_preprocess_returns_subject_stats_dict_for_reuse_by_feature_extraction(tmp_path):
    """
    src/pipeline.py passes result.subject_stats into FeatureExtractor.extract()
    so the two stages don't redundantly recompute identical stats from the
    same split_result - see src/features/extractor.py's module docstring.
    """
    subjects = build_simple_subjects(n_subjects=2, n_channels=3)
    split_result = repetition_split(subjects, test_repetitions=[2])

    result = make_preprocessor(tmp_path).preprocess(subjects, split_result)

    assert set(result.subject_stats.keys()) == {"S0", "S1"}
    for subject_id, stats in result.subject_stats.items():
        assert stats.subject_id == subject_id
        assert stats.channel_mean.shape == (3,)
        assert stats.channel_std.shape == (3,)


def test_preprocess_summary_values_are_native_python_types(tmp_path):
    subjects = build_simple_subjects()
    split_result = repetition_split(subjects, test_repetitions=[2])

    result = make_preprocessor(tmp_path).preprocess(subjects, split_result)

    round_tripped = json.loads(json.dumps(result.summary))
    for key, value in result.summary.items():
        assert type(round_tripped[key]) is type(value)
