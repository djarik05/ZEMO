"""
Dataset Context Builder for Zemo AI Chatbot.
Extracts compact, accurate, grounded summaries of uploaded datasets and experiments
for Gemini context injection without unnecessary token bloat.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select, desc
from core.database.config import get_database
from core.database.models.dataset import Dataset
from core.database.models.experiment import Experiment
from services.ml.analyzer import DatasetAnalyzer, get_dataset_analyzer
from services.ml.loader import DatasetLoader, get_dataset_loader

logger = logging.getLogger(__name__)


class DatasetContextBuilder:
    """Builds structured, lightweight dataset context for the AI Chatbot."""

    def __init__(
        self,
        loader: Optional[DatasetLoader] = None,
        analyzer: Optional[DatasetAnalyzer] = None,
    ):
        self.loader = loader or get_dataset_loader()
        self.analyzer = analyzer or get_dataset_analyzer()

    async def build_context_summary(self, dataset_id: Optional[str]) -> str:
        """
        Produce a concise, factual summary of the active dataset and any trained models.
        """
        if not dataset_id:
            return ""

        try:
            df, version = await self.loader.load_by_dataset_id(dataset_id)
        except Exception as e:
            logger.debug("Could not load dataset %s for chatbot context: %s", dataset_id, e)
            return ""

        if df is None or df.empty:
            return ""

        # Fetch basic dataset metadata
        db = get_database()
        dataset_name = "Uploaded Dataset"
        dataset_format = "CSV"
        async with db.session() as session:
            ds_rec = await session.get(Dataset, dataset_id)
            if ds_rec:
                dataset_name = ds_rec.name or ds_rec.original_filename
                dataset_format = str(ds_rec.format).upper()

            # Check if there are any experiments for this dataset
            exp_stmt = (
                select(Experiment)
                .where(Experiment.dataset_id == dataset_id)
                .order_by(desc(Experiment.created_at))
                .limit(3)
            )
            exp_res = await session.execute(exp_stmt)
            experiments = exp_res.scalars().all()

        # Run fast profile analysis
        analysis = self.analyzer.analyze_dataset(df)
        columns = analysis.get("columns", [])
        total_rows = analysis.get("total_rows", len(df))
        total_columns = analysis.get("total_columns", len(df.columns))

        # Categorize columns
        num_cols = [c["name"] for c in columns if c.get("inferred_type") in ("numeric", "integer", "float")]
        cat_cols = [c["name"] for c in columns if c.get("inferred_type") in ("categorical", "boolean")]
        dt_cols = [c["name"] for c in columns if c.get("inferred_type") == "datetime"]
        id_cols = [c["name"] for c in columns if c.get("inferred_type") == "identifier"]

        # Missing values summary
        cols_with_missing = [c for c in columns if c.get("missing_count", 0) > 0]
        missing_lines = []
        if cols_with_missing:
            for c in cols_with_missing[:8]:
                missing_lines.append(f"`{c['name']}` ({c.get('missing_count', 0)} missing, {c.get('missing_percentage', 0.0):.1f}%)")
            missing_summary = ", ".join(missing_lines)
            if len(cols_with_missing) > 8:
                missing_summary += f", and {len(cols_with_missing) - 8} more."
        else:
            missing_summary = "None (Dataset is completely clean with 0 missing cells)"

        # Target candidates
        target_candidates = analysis.get("target_candidates") or []
        candidate_summary = ", ".join([f"`{col}`" for col in target_candidates[:5]]) if target_candidates else "None detected automatically"

        # Recent experiments summary
        exp_summary_lines = []
        for exp in experiments:
            metrics_snippet = ""
            if exp.metrics:
                if exp.task == "classification":
                    acc = exp.metrics.get("accuracy")
                    f1 = exp.metrics.get("f1_weighted") or exp.metrics.get("f1_score")
                    metrics_snippet = f"Accuracy: {acc:.2f}, F1: {f1:.2f}" if acc is not None else ""
                elif exp.task == "regression":
                    r2 = exp.metrics.get("r2") or exp.metrics.get("r2_score")
                    rmse = exp.metrics.get("rmse")
                    metrics_snippet = f"R²: {r2:.2f}, RMSE: {rmse:.2f}" if r2 is not None else ""
            status_text = f"Status: {exp.status}"
            if metrics_snippet:
                status_text += f" ({metrics_snippet})"
            exp_summary_lines.append(f"- Model `{exp.name}`: Algorithm `{exp.algorithm_id}`, Task: `{exp.task}`, {status_text}")

        # Assemble compact summary
        parts = [
            f"**Dataset Name**: {dataset_name} ({dataset_format})",
            f"**Dimensions**: {total_rows:,} rows × {total_columns} columns",
            f"**Numeric Columns** ({len(num_cols)}): {', '.join([f'`{c}`' for c in num_cols[:15]]) or 'None'}",
            f"**Categorical Columns** ({len(cat_cols)}): {', '.join([f'`{c}`' for c in cat_cols[:15]]) or 'None'}",
        ]

        if id_cols:
            parts.append(f"**Potential Identifier/ID Columns** (recommended to exclude): {', '.join([f'`{c}`' for c in id_cols])}")

        parts.append(f"**Missing Values**: {missing_summary}")
        parts.append(f"**Potential Target Column Candidates**: {candidate_summary}")

        if exp_summary_lines:
            parts.append("\n**Trained Experiments on this Dataset**:\n" + "\n".join(exp_summary_lines))

        return "\n".join(parts)


_context_builder: Optional[DatasetContextBuilder] = None


def get_dataset_context_builder() -> DatasetContextBuilder:
    global _context_builder
    if _context_builder is None:
        _context_builder = DatasetContextBuilder()
    return _context_builder
