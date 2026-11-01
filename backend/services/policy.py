from datetime import datetime
from typing import Any, Optional, List

from core.config import get_settings
from core.database.config import get_db_session
from core.database.models.dataset import (
    Dataset,
    Scan,
    Finding,
    Policy,
    PolicyRule,
    PolicyEvaluation,
    PolicyDecision,
    PolicyDecisionType,
    PolicyRuleCategory,
    ApprovalRecord,
    FindingCategory,
    FindingSeverity,
    FindingConfidence,
    ScanStatus,
)
from core.database.models.dataset_version import DatasetVersion
from core.exceptions import NotFoundError, AppValidationError as ValidationError
from core.logging import get_logger

logger = get_logger(__name__)


class PolicyEvaluationService:
    def __init__(self):
        self.settings = get_settings()

    async def evaluate_dataset(
        self,
        dataset_id: str,
        policy_id: str,
        version_id: Optional[str] = None,
    ) -> PolicyEvaluation:
        """Evaluate a dataset against a policy."""
        async for session in get_db_session():
            # Get policy
            policy = await session.get(Policy, policy_id)
            if not policy:
                raise NotFoundError(f"Policy not found: {policy_id}")
            if not policy.is_active:
                raise ValidationError("Policy is not active")

            # Get dataset
            dataset = await session.get(Dataset, dataset_id)
            if not dataset:
                raise NotFoundError(f"Dataset not found: {dataset_id}")

            # Get version
            if version_id:
                version = await session.get(DatasetVersion, version_id)
                if not version or version.dataset_id != dataset_id:
                    raise NotFoundError(f"Version {version_id} not found for dataset {dataset_id}")
            else:
                from sqlalchemy import select
                version = await session.execute(
                    select(DatasetVersion)
                    .where(DatasetVersion.dataset_id == dataset_id)
                    .order_by(DatasetVersion.version_number.desc())
                    .limit(1)
                )
                version = version.scalar_one_or_none()
                if not version:
                    raise NotFoundError(f"No version found for dataset: {dataset_id}")

            # Get latest completed scan for this version
            from sqlalchemy import select
            scan_result = await session.execute(
                select(Scan)
                .where(Scan.version_id == version.id)
                .where(Scan.status == ScanStatus.COMPLETED)
                .order_by(Scan.completed_at.desc())
                .limit(1)
            )
            scan = scan_result.scalar_one_or_none()
            if not scan:
                raise ValidationError("Dataset version must have a completed scan before policy evaluation")

            # Get active rules for this policy
            active_rules_result = await session.execute(
                select(PolicyRule)
                .where(PolicyRule.policy_id == policy.id)
                .where(PolicyRule.is_active == True)
                .order_by(PolicyRule.priority.desc())
            )
            active_rules = active_rules_result.scalars().all()

            # Get all findings for this scan
            findings_result = await session.execute(
                select(Finding).where(Finding.scan_id == scan.id)
            )
            findings = findings_result.scalars().all()

            # Evaluate rules against findings
            matched_rules = []
            decisions = []
            overall_decision = PolicyDecisionType.ALLOW
            requires_approval = False
            reasons = []

            for rule in active_rules:
                matched = self._evaluate_rule(rule, findings, dataset, version)
                if matched:
                    matched_rules.append({
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "action": rule.action.value,
                        "priority": rule.priority,
                    })
                    
                    # Determine decision based on rule action
                    if rule.action == PolicyDecisionType.BLOCK:
                        overall_decision = PolicyDecisionType.BLOCK
                        requires_approval = True
                        reasons.append(f"Blocked by rule '{rule.name}': {rule.description}")
                    elif rule.action == PolicyDecisionType.SANITIZE:
                        if overall_decision != PolicyDecisionType.BLOCK:
                            overall_decision = PolicyDecisionType.SANITIZE
                            requires_approval = True
                        reasons.append(f"Sanitization required by rule '{rule.name}': {rule.description}")
                    elif rule.action == PolicyDecisionType.REVIEW:
                        if overall_decision not in (PolicyDecisionType.BLOCK, PolicyDecisionType.SANITIZE):
                            overall_decision = PolicyDecisionType.REVIEW
                            requires_approval = True
                        reasons.append(f"Review required by rule '{rule.name}': {rule.description}")

            # If no rules matched and no blockers, allow by default
            if not matched_rules:
                overall_decision = PolicyDecisionType.ALLOW
                reasons.append("No policy rules matched; defaulting to ALLOW")

            # Check if approval is required based on policy default action
            if not requires_approval and policy.default_action in (PolicyDecisionType.REVIEW, PolicyDecisionType.SANITIZE, PolicyDecisionType.BLOCK):
                requires_approval = True

            # Create evaluation record
            evaluation = PolicyEvaluation(
                policy_id=policy.id,
                dataset_id=dataset.id,
                version_id=version.id,
                scan_id=scan.id,
                policy_version=policy.version,
                decision=overall_decision,
                reason="; ".join(reasons) if reasons else "No specific reason",
                details={
                    "policy_name": policy.name,
                    "policy_version": policy.version,
                    "scan_id": scan.id,
                    "dataset_name": dataset.name,
                    "version_number": version.version_number,
                    "total_findings": len(findings),
                    "matched_rules_count": len(matched_rules),
                },
                matched_rules=matched_rules,
                requires_approval=requires_approval,
            )
            session.add(evaluation)
            await session.commit()
            await session.refresh(evaluation)

            logger.info("policy_evaluation_completed", evaluation_id=evaluation.id, decision=overall_decision.value)
            return evaluation

    def _evaluate_rule(self, rule: Any, findings: list, dataset: Any, version: Any) -> bool:
        """Evaluate a single policy rule against findings."""
        condition = rule.condition
        
        # Handle different condition types
        if "finding_categories" in condition:
            required_categories = condition["finding_categories"]
            if not all(any(f.category.value == cat for f in findings) for cat in required_categories):
                return False
        
        if "min_severity" in condition:
            severity_order = {"INFORMATIONAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            min_severity = condition["min_severity"]
            min_level = severity_order.get(min_severity, 0)
            if not any(severity_order.get(f.severity.value, 0) >= min_level for f in findings):
                return False
        
        if "min_confidence" in condition:
            confidence_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            min_conf = condition["min_confidence"]
            min_level = confidence_order.get(min_conf, 0)
            if not any(confidence_order.get(f.confidence.value, 0) >= min_level for f in findings):
                return False
        
        if "finding_categories_any" in condition:
            any_categories = condition["finding_categories_any"]
            if not any(f.category.value in any_categories for f in findings):
                return False
        
        if "max_findings" in condition:
            if len(findings) > condition["max_findings"]:
                return False
        
        return True

    async def get_evaluation(self, evaluation_id: str) -> PolicyEvaluation:
        async for session in get_db_session():
            evaluation = await session.get(PolicyEvaluation, evaluation_id)
            if not evaluation:
                raise NotFoundError(f"Policy evaluation not found: {evaluation_id}")
            return evaluation

    async def list_evaluations(
        self,
        dataset_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[PolicyEvaluation], int]:
        from sqlalchemy import select, func
        async for session in get_db_session():
            query = select(PolicyEvaluation)
            if dataset_id:
                query = query.where(PolicyEvaluation.dataset_id == dataset_id)
            if policy_id:
                query = query.where(PolicyEvaluation.policy_id == policy_id)

            total_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(total_query)
            total = total_result.scalar() or 0

            query = query.order_by(PolicyEvaluation.created_at.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            result = await session.execute(query)
            evaluations = result.scalars().all()

            return list(evaluations), total

    async def approve_evaluation(
        self,
        evaluation_id: str,
        approved_by: str,
        decision: PolicyDecisionType,
        comment: Optional[str] = None,
    ) -> PolicyEvaluation:
        async for session in get_db_session():
            evaluation = await session.get(PolicyEvaluation, evaluation_id)
            if not evaluation:
                raise NotFoundError(f"Policy evaluation not found: {evaluation_id}")

            if not evaluation.requires_approval:
                raise ValidationError("This evaluation does not require approval")

            if evaluation.approved_by:
                raise ValidationError("Evaluation has already been approved")

            # Update evaluation
            evaluation.decision = decision
            evaluation.approved_by = approved_by
            evaluation.approved_at = datetime.utcnow()
            evaluation.completed_at = datetime.utcnow()

            # Create approval record
            approval = ApprovalRecord(
                evaluation_id=evaluation.id,
                approved_by=approved_by,
                decision=decision,
                comment=comment,
            )
            session.add(approval)

            await session.commit()
            await session.refresh(evaluation)

            logger.info("policy_evaluation_approved", evaluation_id=evaluation_id, decision=decision.value, approved_by=approved_by)
            return evaluation

    async def create_policy(
        self,
        name: str,
        description: Optional[str],
        intended_uses: list[str],
        default_action: PolicyDecisionType,
        created_by: Optional[str] = None,
    ) -> Policy:
        async for session in get_db_session():
            # Check for duplicate name
            from sqlalchemy import select
            existing = await session.execute(select(Policy).where(Policy.name == name))
            if existing.scalar_one_or_none():
                raise ValidationError(f"Policy with name '{name}' already exists")

            policy = Policy(
                name=name,
                description=description,
                intended_uses=intended_uses,
                default_action=default_action,
                created_by=created_by,
            )
            session.add(policy)
            await session.commit()
            await session.refresh(policy)

            logger.info("policy_created", policy_id=policy.id, name=name)
            return policy

    async def create_policy_rule(
        self,
        policy_id: str,
        name: str,
        description: Optional[str],
        category: str,
        condition: dict,
        action: str,
        priority: int = 0,
    ) -> PolicyRule:
        async for session in get_db_session():
            policy = await session.get(Policy, policy_id)
            if not policy:
                raise NotFoundError(f"Policy not found: {policy_id}")

            rule = PolicyRule(
                policy_id=policy_id,
                name=name,
                description=description,
                category=PolicyRuleCategory(category),
                condition=condition,
                action=PolicyDecisionType(action),
                priority=priority,
            )
            session.add(rule)
            await session.commit()
            await session.refresh(rule)

            logger.info("policy_rule_created", rule_id=rule.id, policy_id=policy_id, name=name)
            return rule

    async def get_policy(self, policy_id: str) -> Policy:
        async for session in get_db_session():
            policy = await session.get(Policy, policy_id)
            if not policy:
                raise NotFoundError(f"Policy not found: {policy_id}")
            return policy

    async def list_policies(self, active_only: bool = True) -> list[Policy]:
        from sqlalchemy import select
        async for session in get_db_session():
            query = select(Policy)
            if active_only:
                query = query.where(Policy.is_active == True)
            query = query.order_by(Policy.name)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def update_policy(
        self,
        policy_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        intended_uses: Optional[list[str]] = None,
        default_action: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Policy:
        async for session in get_db_session():
            policy = await session.get(Policy, policy_id)
            if not policy:
                raise NotFoundError(f"Policy not found: {policy_id}")

            if name is not None:
                policy.name = name
            if description is not None:
                policy.description = description
            if intended_uses is not None:
                policy.intended_uses = intended_uses
            if default_action is not None:
                policy.default_action = PolicyDecisionType(default_action)
            if is_active is not None:
                policy.is_active = is_active

            policy.updated_at = datetime.utcnow()
            await session.commit()
            await session.refresh(policy)

            logger.info("policy_updated", policy_id=policy_id)
            return policy

    async def delete_policy(self, policy_id: str) -> bool:
        async for session in get_db_session():
            policy = await session.get(Policy, policy_id)
            if not policy:
                raise NotFoundError(f"Policy not found: {policy_id}")

            # Check if policy has evaluations
            from sqlalchemy import select
            evaluations = await session.execute(select(PolicyEvaluation).where(PolicyEvaluation.policy_id == policy_id))
            if evaluations.scalar_one_or_none():
                raise ValidationError("Cannot delete policy with existing evaluations")

            await session.delete(policy)
            await session.commit()

            logger.info("policy_deleted", policy_id=policy_id)
            return True


def get_policy_evaluation_service() -> PolicyEvaluationService:
    return PolicyEvaluationService()