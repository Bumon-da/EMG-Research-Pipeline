import json

import pandas as pd

from src.dashboard import data_access


def _make_run(tmp_path, name="experiment_20260101_000000"):
    run_path = tmp_path / name
    (run_path / "validation").mkdir(parents=True)
    (run_path / "eda").mkdir(parents=True)
    return data_access.ExperimentRun(name=name, path=run_path)


def test_list_experiment_runs_sorted_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "EXPERIMENTS_DIR", tmp_path)

    for name in ["experiment_20260101_000000", "experiment_20260301_000000", "experiment_20260201_000000"]:
        (tmp_path / name).mkdir()

    runs = data_access.list_experiment_runs()

    assert [r.name for r in runs] == [
        "experiment_20260301_000000",
        "experiment_20260201_000000",
        "experiment_20260101_000000",
    ]


def test_list_experiment_runs_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, "EXPERIMENTS_DIR", tmp_path / "does_not_exist")

    assert data_access.list_experiment_runs() == []


def test_load_manifest_missing_returns_empty_dict(tmp_path):
    run = _make_run(tmp_path)

    assert data_access.load_manifest(run) == {}


def test_load_manifest_reads_json(tmp_path):
    run = _make_run(tmp_path)
    (run.path / "manifest.json").write_text(json.dumps({"experiment_name": run.name}))

    manifest = data_access.load_manifest(run)

    assert manifest["experiment_name"] == run.name


def test_load_validation_details_missing_returns_empty_dataframe(tmp_path):
    run = _make_run(tmp_path)

    df = data_access.load_validation_details(run)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_class_distribution_reads_csv(tmp_path):
    run = _make_run(tmp_path)
    csv_path = run.validation_dir / "class_distribution.csv"
    csv_path.write_text("Exercise,Label,Samples\n1,0,100\n1,1,50\n")

    df = data_access.load_class_distribution(run)

    assert list(df["Samples"]) == [100, 50]


def test_load_integrity_report_missing_returns_empty_string(tmp_path):
    run = _make_run(tmp_path)

    assert data_access.load_integrity_report(run) == ""
