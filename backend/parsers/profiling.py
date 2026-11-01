from collections import Counter
from typing import Any

import pandas as pd
import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


def infer_column_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"

    sample_size = min(1000, len(non_null))
    sample = non_null.sample(n=sample_size, random_state=42) if len(non_null) > sample_size else non_null

    numeric_count = 0
    integer_count = 0
    boolean_count = 0
    datetime_count = 0

    for val in sample:
        if isinstance(val, bool):
            boolean_count += 1
        elif isinstance(val, (int, np.integer)):
            integer_count += 1
            numeric_count += 1
        elif isinstance(val, (float, np.floating)):
            numeric_count += 1
        elif isinstance(val, str):
            try:
                float(val)
                numeric_count += 1
                if float(val).is_integer():
                    integer_count += 1
            except ValueError:
                pass

    total = len(sample)
    if boolean_count / total > 0.8:
        return "boolean"
    if integer_count / total > 0.8:
        return "integer"
    if numeric_count / total > 0.8:
        return "numeric"
    if datetime_count / total > 0.8:
        return "datetime"
    return "categorical"


def calculate_profile(records: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
    if not records:
        return {
            "record_count": 0,
            "column_count": 0,
            "columns": [],
            "schema": {},
            "statistics": {},
            "duplicate_records": 0,
            "empty_columns": [],
        }

    df = pd.DataFrame(records)

    schema = {}
    statistics = {}
    empty_columns = []

    for col in columns:
        series = df[col] if col in df.columns else pd.Series([None] * len(df))
        non_null = series.dropna()
        null_count = series.isna().sum() + (series == "").sum()
        null_pct = (null_count / len(series) * 100) if len(series) > 0 else 100

        col_type = infer_column_type(series)

        col_schema = {
            "name": col,
            "type": col_type,
            "null_count": int(null_count),
            "null_percentage": round(null_pct, 2),
        }

        if col_type in ("integer", "numeric"):
            try:
                numeric_series = pd.to_numeric(non_null, errors="coerce").dropna()
                if not numeric_series.empty:
                    col_schema.update({
                        "min": float(numeric_series.min()),
                        "max": float(numeric_series.max()),
                        "mean": float(numeric_series.mean()),
                        "std": float(numeric_series.std()),
                    })
                    statistics[col] = {
                        "min": float(numeric_series.min()),
                        "max": float(numeric_series.max()),
                        "mean": float(numeric_series.mean()),
                        "std": float(numeric_series.std()),
                        "median": float(numeric_series.median()),
                        "q1": float(numeric_series.quantile(0.25)),
                        "q3": float(numeric_series.quantile(0.75)),
                    }
            except Exception:
                pass
        elif col_type == "categorical":
            value_counts = non_null.value_counts()
            unique_count = len(value_counts)
            col_schema["unique_count"] = int(unique_count)
            col_schema["top_values"] = value_counts.head(10).to_dict()
            statistics[col] = {
                "unique_count": int(unique_count),
                "top_values": value_counts.head(10).to_dict(),
            }
            if unique_count == 0:
                empty_columns.append(col)
        elif col_type == "boolean":
            value_counts = non_null.value_counts()
            col_schema["distribution"] = value_counts.to_dict()
            statistics[col] = {"distribution": value_counts.to_dict()}

        schema[col] = col_schema

    df_for_dup = df[columns] if columns else df
    duplicate_records = int(df_for_dup.duplicated().sum())

    profile = {
        "record_count": len(records),
        "column_count": len(columns),
        "columns": columns,
        "schema": schema,
        "statistics": statistics,
        "duplicate_records": duplicate_records,
        "empty_columns": empty_columns,
    }

    logger.info("profile_calculated", record_count=len(records), column_count=len(columns))
    return profile