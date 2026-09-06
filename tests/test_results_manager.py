import pandas as pd

from src.managers.results_manager import ResultsManager


def test_save_parquet_roundtrips_dataframe(tmp_path):
    results = ResultsManager(output_root=tmp_path)

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    filepath = results.save_parquet(dataframe=df, folder="features", filename="test.parquet")

    assert filepath.exists()
    assert filepath.suffix == ".parquet"

    loaded = pd.read_parquet(filepath)
    pd.testing.assert_frame_equal(loaded, df)


def test_save_parquet_index_true_preserves_named_index(tmp_path):
    """
    Unlike save_csv (where index=True writes the index as a plain column
    that read_csv reads back as a column), Parquet round-trips a named
    index as an actual index - assert that behavior rather than assuming
    CSV's flattening semantics apply here.
    """
    results = ResultsManager(output_root=tmp_path)

    df = pd.DataFrame({"a": [1, 2]}, index=["r1", "r2"])
    df.index.name = "row_id"

    filepath = results.save_parquet(dataframe=df, folder="features", filename="indexed.parquet", index=True)

    loaded = pd.read_parquet(filepath)
    assert loaded.index.name == "row_id"
    assert list(loaded.index) == ["r1", "r2"]


def test_save_parquet_index_false_drops_index(tmp_path):
    results = ResultsManager(output_root=tmp_path)

    df = pd.DataFrame({"a": [1, 2]}, index=["r1", "r2"])
    df.index.name = "row_id"

    filepath = results.save_parquet(dataframe=df, folder="features", filename="no_index.parquet", index=False)

    loaded = pd.read_parquet(filepath)
    assert list(loaded.columns) == ["a"]
    assert loaded.index.name is None
