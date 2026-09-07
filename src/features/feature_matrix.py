"""
Feature Matrix Read Access

The read-side API for a run's feature output - used by v0.7.0's model
training and the dashboard's Features page, so neither hardcodes the
per-subject Parquet layout src/features/extractor.py writes. Deliberately
generic over a plain directory path (not an ExperimentRun) so this module
has no dependency on src/dashboard - the dependency runs the other way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_subject_feature_files(features_dir: Path) -> list[Path]:
    """All per-subject feature Parquet files in a run's features/ folder."""
    if not features_dir.exists():
        return []
    return sorted(features_dir.glob("*_features.parquet"))


def load_subject_features(features_dir: Path, subject_id: str) -> pd.DataFrame:
    """
    Load one subject's feature matrix. Returns an empty DataFrame if the
    file doesn't exist, matching src/dashboard/data_access.py's
    missing-file convention.
    """
    filepath = features_dir / f"{subject_id}_features.parquet"
    if not filepath.exists():
        return pd.DataFrame()
    return pd.read_parquet(filepath)


def load_all_features(features_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate every subject's feature matrix in a run. This
    materializes the full matrix (~150-175MB at float32/int8) - callers
    that only need one subject or one exercise should filter after
    loading a single file instead of calling this, where practical.
    """
    files = list_subject_feature_files(features_dir)
    if not files:
        return pd.DataFrame()

    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """split: 'train' or 'test'."""
    return df[df["Split"] == split]


def filter_exercise(df: pd.DataFrame, exercise: int) -> pd.DataFrame:
    return df[df["Exercise"] == exercise]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Every column that isn't metadata - what a model trains on. Mirrors
    the metadata column set src/features/extractor.py writes.
    """
    metadata = {"Subject", "Trial", "Exercise", "Label", "GlobalLabel", "Split", "Start"}
    return [c for c in df.columns if c not in metadata]
