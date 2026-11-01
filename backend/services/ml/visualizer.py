from typing import Any, Optional
import numpy as np
import pandas as pd


class VisualizationEngine:
    """
    Computes structured visualization payloads for histograms, box plots,
    categorical distributions, missing-value profiles, correlation matrices, and scatter plots.
    """

    def generate_all_visualizations(
        self,
        df: pd.DataFrame,
        selected_numeric: Optional[str] = None,
        selected_categorical: Optional[str] = None,
        scatter_x: Optional[str] = None,
        scatter_y: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate complete visualization package for frontend charts.
        """
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        ]
        categorical_cols = [
            c for c in df.columns
            if c not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(df[c])
        ]

        # 1. Histograms for numeric columns
        histograms: dict[str, Any] = {}
        target_num_cols = [selected_numeric] if selected_numeric and selected_numeric in numeric_cols else numeric_cols[:10]
        for col in target_num_cols:
            histograms[col] = self._compute_histogram(df[col], bins_count=8)

        # 2. Box plots for numeric columns
        box_plots: dict[str, Any] = {}
        for col in target_num_cols:
            box_plots[col] = self._compute_box_plot(df[col])

        # 3. Categorical distributions
        categories: dict[str, Any] = {}
        target_cat_cols = [selected_categorical] if selected_categorical and selected_categorical in categorical_cols else categorical_cols[:10]
        for col in target_cat_cols:
            categories[col] = self._compute_categorical_distribution(df[col], top_n=8)

        # 4. Missing value overview
        missing_overview = self._compute_missing_overview(df)

        # 5. Correlation matrix (numeric columns, max 15 for clean UI)
        correlation_matrix = self._compute_correlation_matrix(df, numeric_cols[:15])

        # 6. Scatter plot
        scatter_data = self._compute_scatter_plot(df, numeric_cols, scatter_x, scatter_y)

        # Default active keys for instant UI rendering
        primary_numeric = selected_numeric if selected_numeric in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        primary_categorical = selected_categorical if selected_categorical in categorical_cols else (categorical_cols[0] if categorical_cols else None)

        return {
            "primary_numeric": primary_numeric,
            "primary_categorical": primary_categorical,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "histograms": histograms,
            "box_plots": box_plots,
            "categories": categories,
            "missing_overview": missing_overview,
            "correlation_matrix": correlation_matrix,
            "scatter_plot": scatter_data,
        }

    def _compute_histogram(self, series: pd.Series, bins_count: int = 8) -> dict[str, Any]:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) == 0:
            return {"labels": [], "counts": [], "min": 0, "max": 0, "bin_width": 0}

        min_val = float(clean.min())
        max_val = float(clean.max())

        if min_val == max_val:
            return {
                "labels": [f"{min_val:.1f}"],
                "counts": [len(clean)],
                "min": min_val,
                "max": max_val,
                "bin_width": 0,
            }

        counts, bin_edges = np.histogram(clean, bins=bins_count)
        labels = [
            f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}"
            for i in range(len(counts))
        ]

        return {
            "labels": labels,
            "counts": [int(c) for c in counts],
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "bin_width": round(float(bin_edges[1] - bin_edges[0]), 2),
        }

    def _compute_box_plot(self, series: pd.Series) -> dict[str, Any]:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) == 0:
            return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0, "outliers": []}

        q1 = float(clean.quantile(0.25))
        median = float(clean.median())
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        non_outliers = clean[(clean >= lower_bound) & (clean <= upper_bound)]
        outliers = clean[(clean < lower_bound) | (clean > upper_bound)]

        return {
            "min": round(float(non_outliers.min()) if len(non_outliers) > 0 else float(clean.min()), 2),
            "q1": round(q1, 2),
            "median": round(median, 2),
            "q3": round(q3, 2),
            "max": round(float(non_outliers.max()) if len(non_outliers) > 0 else float(clean.max()), 2),
            "iqr": round(iqr, 2),
            "outlier_count": len(outliers),
            "outliers": [round(float(v), 2) for v in outliers.head(30).tolist()],
        }

    def _compute_categorical_distribution(self, series: pd.Series, top_n: int = 8) -> dict[str, Any]:
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return {"categories": [], "counts": [], "other_count": 0}

        counts = clean.value_counts()
        top_cats = counts.head(top_n)
        other_sum = int(counts.iloc[top_n:].sum()) if len(counts) > top_n else 0

        return {
            "categories": [str(c) for c in top_cats.index],
            "counts": [int(c) for c in top_cats.values],
            "other_count": other_sum,
            "total_categories": len(counts),
        }

    def _compute_missing_overview(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        total_rows = len(df)
        missing_data = []

        for col in df.columns:
            null_count = int(df[col].isna().sum())
            null_pct = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
            missing_data.append({
                "column": str(col),
                "missing_count": null_count,
                "missing_pct": null_pct,
                "present_count": total_rows - null_count,
            })

        # Sort with highest missing percentage first
        missing_data.sort(key=lambda x: x["missing_pct"], reverse=True)
        return missing_data

    def _compute_correlation_matrix(self, df: pd.DataFrame, num_cols: list[str]) -> dict[str, Any]:
        if len(num_cols) < 2:
            return {"columns": num_cols, "matrix": []}

        subset = df[num_cols].apply(pd.to_numeric, errors="coerce")
        corr = subset.corr(method="pearson").fillna(0.0)

        matrix = [
            [round(float(val), 3) for val in row]
            for row in corr.values
        ]

        return {
            "columns": list(corr.columns),
            "matrix": matrix,
        }

    def _compute_scatter_plot(
        self,
        df: pd.DataFrame,
        num_cols: list[str],
        x_col: Optional[str],
        y_col: Optional[str],
    ) -> dict[str, Any]:
        if len(num_cols) < 2:
            return {"x_col": None, "y_col": None, "points": []}

        col_x = x_col if x_col and x_col in num_cols else num_cols[0]
        col_y = y_col if y_col and y_col in num_cols and y_col != col_x else (num_cols[1] if len(num_cols) > 1 else num_cols[0])

        subset = df[[col_x, col_y]].dropna()
        if len(subset) > 500:
            subset = subset.sample(n=500, random_state=42)

        points = [
            {"x": round(float(r[col_x]), 3), "y": round(float(r[col_y]), 3)}
            for _, r in subset.iterrows()
        ]

        return {
            "x_col": col_x,
            "y_col": col_y,
            "points": points,
            "total_sampled": len(points),
        }


_visualization_engine: Optional[VisualizationEngine] = None


def get_visualization_engine() -> VisualizationEngine:
    global _visualization_engine
    if _visualization_engine is None:
        _visualization_engine = VisualizationEngine()
    return _visualization_engine
