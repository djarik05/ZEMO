import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from core.exceptions import NotFoundError, AppValidationError
from services.ml.loader import DatasetLoader


@pytest.fixture
def loader():
    return DatasetLoader()


def test_load_csv_numeric_inference(loader):
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        f.write("id,age,income,is_active,category\n")
        f.write("1,25,50000.50,true,sales\n")
        f.write("2,30,60000.00,false,engineering\n")
        f.write("3,45,120000.75,true,management\n")
        f_path = f.name

    try:
        df = loader.load_from_path(f_path, "csv")
        assert len(df) == 3
        assert list(df.columns) == ["id", "age", "income", "is_active", "category"]
        assert pd.api.types.is_numeric_dtype(df["age"])
        assert pd.api.types.is_float_dtype(df["income"])
        assert df["income"].iloc[0] == 50000.50
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_load_csv_na_inference(loader):
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        f.write("col_a,col_b\n")
        f.write("10,normal\n")
        f.write("NA,null\n")
        f.write("30,missing_val\n")
        f_path = f.name

    try:
        df = loader.load_from_path(f_path, "csv")
        assert len(df) == 3
        assert pd.isna(df["col_a"].iloc[1])
        assert pd.isna(df["col_b"].iloc[1])
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_load_json_and_jsonl(loader):
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
        f.write('{"x": 1.5, "y": 10}\n{"x": 2.5, "y": 20}\n')
        f_path = f.name

    try:
        df = loader.load_from_path(f_path, "jsonl")
        assert len(df) == 2
        assert df["x"].sum() == 4.0
        assert df["y"].sum() == 30
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_load_excel(loader):
    with tempfile.NamedTemporaryFile("wb", suffix=".xlsx", delete=False) as f:
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
        test_df.to_excel(f.name, index=False)
        f_path = f.name

    try:
        df = loader.load_from_path(f_path, "xlsx")
        assert len(df) == 3
        assert list(df.columns) == ["col1", "col2"]
        assert df["col1"].sum() == 6
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_load_parquet(loader):
    with tempfile.NamedTemporaryFile("wb", suffix=".parquet", delete=False) as f:
        test_df = pd.DataFrame({"p1": [10.5, 20.5], "p2": ["x", "y"]})
        test_df.to_parquet(f.name, index=False)
        f_path = f.name

    try:
        df = loader.load_from_path(f_path, "parquet")
        assert len(df) == 2
        assert list(df.columns) == ["p1", "p2"]
        assert df["p1"].sum() == 31.0
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_load_nonexistent_file(loader):
    with pytest.raises(NotFoundError):
        loader.load_from_path("/non/existent/path.csv", "csv")


def test_load_unsupported_format(loader):
    with tempfile.NamedTemporaryFile("w+", suffix=".wav", delete=False) as f:
        f_path = f.name

    try:
        with pytest.raises(AppValidationError):
            loader.load_from_path(f_path, "wav")
    finally:
        Path(f_path).unlink(missing_ok=True)

