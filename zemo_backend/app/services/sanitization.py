"""
Dataset sanitization and redaction service.

Applies masking, redaction, hashing, and record removal to sensitive findings.
Creates a new sanitized dataset version while leaving the original dataset unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from app.core.exceptions import DatasetNotFoundError, SanitizationError
from app.core.logging import get_logger
from app.core.security import generate_storage_id
from app.models.enums import (
    DatasetStatus,
    FindingCategory,
    PolicyDecision,
    RemediationStatus,
    Severity,
    utcnow,
)
from app.repositories.dataset_repo import DatasetRepository
from app.schemas.sanitization import (
    SanitizationAction,
    SanitizationRecord,
    SanitizationRequest,
    SanitizationResult,
)
from app.utils.masking import (
    hash_value,
    mask_credit_card,
    mask_email,
    mask_generic,
    mask_phone,
    redact_value,
)

logger = get_logger("sanitization")


class SanitizationService:
    """Service to sanitize datasets by applying masking, redaction, or record removal."""

    def __init__(self, repo: DatasetRepository | None = None) -> None:
        self.repo = repo or DatasetRepository()

    def sanitize(
        self,
        dataset_id: str,
        actions: list[SanitizationAction] | None = None,
        auto_apply_policy: bool = True,
    ) -> SanitizationResult:
        """
        Sanitize dataset:
        1. Fetch original dataframe and findings.
        2. Determine sanitization actions (explicit or automatic from policy).
        3. Apply transformations to a deep copy of the DataFrame.
        4. Persist newly sanitized dataset with new dataset_id.
        5. Return SanitizationResult. Original dataset remains untouched.
        """
        df = self.repo.get_dataframe(dataset_id)
        if df is None:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' not found.",
                detail={"dataset_id": dataset_id},
            )

        metadata = self.repo.get_metadata(dataset_id)
        if metadata is None:
            raise DatasetNotFoundError(
                f"Metadata for dataset '{dataset_id}' not found.",
                detail={"dataset_id": dataset_id},
            )

        findings = self.repo.get_findings(dataset_id)
        orig_record_count = len(df)
        df_clean = df.copy(deep=True)

        records_to_remove: set[int] = set()
        fields_redacted = 0
        fields_masked = 0
        fields_hashed = 0
        actions_applied: list[SanitizationRecord] = []

        # Index findings by finding_id for lookup
        finding_map = {f["finding_id"]: f for f in findings}

        # Build list of action items: explicit or automatic
        active_actions: list[SanitizationAction] = []
        if actions:
            active_actions.extend(actions)
        elif auto_apply_policy and findings:
            for f in findings:
                cat = f.get("category", "")
                sev = f.get("severity", "")
                f_id = f.get("finding_id", "")
                field_name = f.get("field_name")

                if cat in (FindingCategory.PROMPT_INJECTION.value, FindingCategory.SUSPICIOUS_CONTENT.value, FindingCategory.MALFORMED_RECORD.value):
                    active_actions.append(SanitizationAction(
                        finding_id=f_id,
                        action="remove_record",
                        target_field=field_name,
                    ))
                elif cat == FindingCategory.EMAIL.value:
                    active_actions.append(SanitizationAction(
                        finding_id=f_id,
                        action="mask",
                        target_field=field_name,
                    ))
                elif cat == FindingCategory.PHONE.value:
                    active_actions.append(SanitizationAction(
                        finding_id=f_id,
                        action="mask",
                        target_field=field_name,
                    ))
                elif cat in (FindingCategory.CREDIT_CARD.value, FindingCategory.PERSONAL_ID.value):
                    active_actions.append(SanitizationAction(
                        finding_id=f_id,
                        action="mask",
                        target_field=field_name,
                    ))
                elif cat in (
                    FindingCategory.API_KEY.value,
                    FindingCategory.PASSWORD.value,
                    FindingCategory.ACCESS_TOKEN.value,
                    FindingCategory.PRIVATE_KEY.value,
                    FindingCategory.DB_CREDENTIAL.value,
                ):
                    active_actions.append(SanitizationAction(
                        finding_id=f_id,
                        action="redact",
                        target_field=field_name,
                    ))
                else:
                    active_actions.append(SanitizationAction(
                        finding_id=f_id,
                        action="mask",
                        target_field=field_name,
                    ))

        # Apply actions
        for act in active_actions:
            finding = finding_map.get(act.finding_id)
            field = act.target_field or (finding.get("field_name") if finding else None)
            record_idx = finding.get("record_index") if finding else None
            cat = finding.get("category", "") if finding else ""

            if act.action == "remove_record":
                if record_idx is not None and record_idx in df_clean.index:
                    records_to_remove.add(record_idx)
                    actions_applied.append(SanitizationRecord(
                        finding_id=act.finding_id,
                        action="remove_record",
                        status=RemediationStatus.APPLIED,
                        field_name=field,
                        message=f"Removed record at index {record_idx}.",
                    ))
            elif act.action in ("mask", "redact", "hash", "replace"):
                if field and field in df_clean.columns:
                    if record_idx is not None and record_idx in df_clean.index:
                        indices_to_modify = [record_idx]
                    else:
                        indices_to_modify = list(df_clean.index)

                    for r_idx in indices_to_modify:
                        val = df_clean.at[r_idx, field]
                        if val is None or pd.isna(val):
                            continue
                        val_str = str(val)

                        if act.action == "mask":
                            if cat == FindingCategory.EMAIL.value or "@" in val_str:
                                df_clean.at[r_idx, field] = mask_email(val_str)
                            elif cat == FindingCategory.PHONE.value:
                                df_clean.at[r_idx, field] = mask_phone(val_str)
                            elif cat == FindingCategory.CREDIT_CARD.value:
                                df_clean.at[r_idx, field] = mask_credit_card(val_str)
                            else:
                                df_clean.at[r_idx, field] = mask_generic(val_str)
                            fields_masked += 1
                        elif act.action == "redact":
                            df_clean.at[r_idx, field] = redact_value(val_str)
                            fields_redacted += 1
                        elif act.action == "hash":
                            df_clean.at[r_idx, field] = hash_value(val_str)
                            fields_hashed += 1
                        elif act.action == "replace":
                            df_clean.at[r_idx, field] = act.replacement
                            fields_redacted += 1

                    actions_applied.append(SanitizationRecord(
                        finding_id=act.finding_id,
                        action=act.action,
                        status=RemediationStatus.APPLIED,
                        field_name=field,
                        message=f"Applied {act.action} on field '{field}'.",
                    ))

        # Drop records marked for removal
        if records_to_remove:
            df_clean = df_clean.drop(index=list(records_to_remove)).reset_index(drop=True)

        records_removed = len(records_to_remove)
        sanitized_record_count = len(df_clean)

        # Create new sanitized dataset
        sanitized_dataset_id = generate_storage_id()
        now = utcnow()

        new_meta = {
            "dataset_id": sanitized_dataset_id,
            "original_filename": f"sanitized_{metadata.get('original_filename', 'data.csv')}",
            "sanitized_filename": f"sanitized_{metadata.get('sanitized_filename', 'data.csv')}",
            "format": metadata.get("format", "csv"),
            "file_size_bytes": 0,
            "record_count": sanitized_record_count,
            "column_count": len(df_clean.columns),
            "columns": list(df_clean.columns),
            "status": DatasetStatus.SANITIZED.value,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "version": int(metadata.get("version", 1)) + 1,
            "parent_dataset_id": dataset_id,
            "storage_key": f"datasets/{sanitized_dataset_id}/data.parquet",
        }

        self.repo.save_dataframe(sanitized_dataset_id, df_clean)
        self.repo.save_metadata(new_meta)

        result = SanitizationResult(
            dataset_id=dataset_id,
            sanitized_dataset_id=sanitized_dataset_id,
            original_record_count=orig_record_count,
            sanitized_record_count=sanitized_record_count,
            records_removed=records_removed,
            fields_redacted=fields_redacted,
            fields_masked=fields_masked,
            fields_hashed=fields_hashed,
            actions_applied=actions_applied,
            sanitized_at=now,
            message="Dataset sanitized successfully.",
        )

        self.repo.save_sanitization(dataset_id, result.model_dump(mode="json"))
        logger.info(
            "Sanitized %s -> %s: removed=%d, redacted=%d, masked=%d",
            dataset_id,
            sanitized_dataset_id,
            records_removed,
            fields_redacted,
            fields_masked,
        )

        return result
