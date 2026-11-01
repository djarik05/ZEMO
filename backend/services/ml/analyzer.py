import re
from typing import Any, Optional
import numpy as np
import pandas as pd

ID_REGEX = re.compile(
    r"(^|_)(id|uuid|guid|key|code|slug|index|number|no|ref|hash|token)($|_)",
    re.IGNORECASE,
)


class DatasetAnalyzer:
    """
    Automatic column type detection, dataset profiling, and descriptive statistics engine.
    Uses Pandas and NumPy for high performance and accurate data science semantics.
    """

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        type_overrides: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Produce comprehensive analysis of the DataFrame.
        """
        overrides = type_overrides or {}
        total_rows = len(df)
        total_columns = len(df.columns)

        if total_rows == 0 or total_columns == 0:
            return {
                "total_rows": total_rows,
                "total_columns": total_columns,
                "missing_cells_total": 0,
                "missing_cells_percentage": 0.0,
                "duplicate_rows": 0,
                "duplicate_rows_percentage": 0.0,
                "memory_usage_bytes": 0,
                "memory_usage_formatted": "0 B",
                "column_types_summary": {},
                "columns": [],
                "target_candidates": [],
            }

        # Global dataset statistics
        null_counts_series = df.isna().sum()
        total_cells = total_rows * total_columns
        missing_cells_total = int(null_counts_series.sum())
        missing_cells_percentage = round((missing_cells_total / total_cells) * 100, 2) if total_cells > 0 else 0.0

        duplicate_rows = int(df.duplicated().sum())
        duplicate_rows_percentage = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

        memory_bytes = int(df.memory_usage(deep=True).sum())
        memory_formatted = self._format_bytes(memory_bytes)

        columns_meta: list[dict[str, Any]] = []
        types_summary: dict[str, int] = {
            "numeric": 0,
            "categorical": 0,
            "boolean": 0,
            "datetime": 0,
            "identifier": 0,
            "text": 0,
        }
        target_candidates: list[str] = []

        for col_name in df.columns:
            series = df[col_name]
            inferred_type = self._detect_column_type(series, col_name)

            # Apply user override if provided
            if col_name in overrides:
                inferred_type = overrides[col_name]

            types_summary[inferred_type] = types_summary.get(inferred_type, 0) + 1

            col_profile = self._profile_column(series, col_name, inferred_type, total_rows)
            columns_meta.append(col_profile)

            # Potential target candidates (non-identifier, reasonable cardinality or numeric)
            if inferred_type in ("numeric", "categorical", "boolean") and col_profile["null_percentage"] < 50.0:
                target_candidates.append(str(col_name))

        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "missing_cells_total": missing_cells_total,
            "missing_cells_percentage": missing_cells_percentage,
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_percentage": duplicate_rows_percentage,
            "memory_usage_bytes": memory_bytes,
            "memory_usage_formatted": memory_formatted,
            "column_types_summary": types_summary,
            "columns": columns_meta,
            "target_candidates": target_candidates,
        }

    def _detect_column_type(self, series: pd.Series, col_name: str) -> str:
        """
        Automatically infer semantic column type.
        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return "categorical"

        n_rows = len(series)
        unique_count = non_null.nunique()

        is_string_or_obj = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

        # 1. Datetime detection
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Check if string column looks like datetime
        if is_string_or_obj and len(non_null) > 0:
            sample = non_null.head(50)
            parsed_dates = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed_dates.notna().mean() >= 0.85:
                return "datetime"

        # 2. Boolean detection
        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        # Check binary integer or binary strings
        if unique_count <= 2:
            unique_vals = set(non_null.unique())
            bool_equivalents = [
                {0, 1},
                {0.0, 1.0},
                {"0", "1"},
                {"true", "false"},
                {"yes", "no"},
                {"y", "n"},
                {"t", "f"},
            ]
            lower_set = {str(v).strip().lower() for v in unique_vals}
            if any(lower_set.issubset(b_set) for b_set in bool_equivalents):
                return "boolean"

        # 3. Identifier detection
        is_id_name = bool(ID_REGEX.search(str(col_name)))
        is_strictly_unique = unique_count == n_rows and n_rows >= 5
        is_high_cardinality = (unique_count / n_rows) >= 0.95 and n_rows >= 20

        if is_id_name and (is_strictly_unique or is_high_cardinality):
            return "identifier"

        if is_strictly_unique and is_string_or_obj:
            return "identifier"

        # 4. Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # Check if string column can be safely converted to numeric
        if is_string_or_obj:
            numeric_parsed = pd.to_numeric(non_null.head(100), errors="coerce")
            if numeric_parsed.notna().mean() >= 0.90:
                return "numeric"

        # 5. Free-form text vs Categorical
        if is_string_or_obj:
            avg_len = non_null.astype(str).str.len().mean()
            if avg_len > 100 or (unique_count > 100 and (unique_count / n_rows) > 0.7):
                return "text"
            return "categorical"

        return "categorical"

    def _profile_column(
        self,
        series: pd.Series,
        col_name: str,
        inferred_type: str,
        total_rows: int,
    ) -> dict[str, Any]:
        """
        Generate detailed statistics and sample values for a single column.
        """
        non_null = series.dropna()
        null_count = int(series.isna().sum())
        null_percentage = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        unique_count = int(non_null.nunique())
        unique_percentage = round((unique_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        # Non-null sample values (up to 5)
        sample_vals = [
            self._to_serializable(v)
            for v in non_null.head(5).tolist()
        ]

        stats: dict[str, Any] = {}

        if inferred_type == "numeric":
            num_series = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(num_series) > 0:
                stats = {
                    "min": float(num_series.min()),
                    "max": float(num_series.max()),
                    "mean": round(float(num_series.mean()), 4),
                    "median": round(float(num_series.median()), 4),
                    "std": round(float(num_series.std()), 4) if len(num_series) > 1 else 0.0,
                    "q25": round(float(num_series.quantile(0.25)), 4),
                    "q75": round(float(num_series.quantile(0.75)), 4),
                }
        elif inferred_type in ("categorical", "text", "identifier"):
            value_counts = non_null.astype(str).value_counts().head(10)
            top_vals = [
                {"value": str(k), "count": int(v)}
                for k, v in value_counts.items()
            ]
            stats = {
                "top_values": top_vals,
                "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            }
        elif inferred_type == "boolean":
            bool_series = non_null.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
            true_count = int(bool_series.sum())
            false_count = int(len(bool_series) - true_count)
            stats = {
                "true_count": true_count,
                "false_count": false_count,
                "true_percentage": round((true_count / len(bool_series)) * 100, 2) if len(bool_series) > 0 else 0.0,
            }
        elif inferred_type == "datetime":
            dt_series = pd.to_datetime(non_null, errors="coerce").dropna()
            if len(dt_series) > 0:
                stats = {
                    "min_date": str(dt_series.min()),
                    "max_date": str(dt_series.max()),
                }

        return {
            "name": str(col_name),
            "inferred_type": inferred_type,
            "dtype": str(series.dtype),
            "is_nullable": null_count > 0,
            "null_count": null_count,
            "null_percentage": null_percentage,
            "unique_count": unique_count,
            "unique_percentage": unique_percentage,
            "sample_values": sample_vals,
            "stats": stats,
        }

    def _to_serializable(self, val: Any) -> Any:
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        return str(val)

    def _format_bytes(self, num_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if num_bytes < 1024.0:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.1f} TB"


_dataset_analyzer: Optional[DatasetAnalyzer] = None


def get_dataset_analyzer() -> DatasetAnalyzer:
    global _dataset_analyzer
    if _dataset_analyzer is None:
        _dataset_analyzer = DatasetAnalyzer()
    return _dataset_analyzer
