import re
from typing import Any

from detectors.base import BaseDetector, DetectionContext, Finding
from core.database.models.dataset import (
    FindingCategory,
    FindingSeverity,
    FindingConfidence,
    PromptInjectionRuleType,
)


class PromptInjectionDetector(BaseDetector):
    name = "prompt_injection"
    version = "1.0.0"
    category = FindingCategory.PROMPT_INJECTION

    OVERRIDE_INSTRUCTION_PATTERNS = [
        re.compile(r"(?i)ignore\s+(?:previous|prior|all)\s+(?:instructions|prompts|rules)"),
        re.compile(r"(?i)disregard\s+(?:previous|prior|all)\s+(?:instructions|prompts|rules)"),
        re.compile(r"(?i)forget\s+(?:previous|prior|all)\s+(?:instructions|prompts|rules)"),
        re.compile(r"(?i)override\s+(?:previous|prior|all)\s+(?:instructions|prompts|rules)"),
        re.compile(r"(?i)new\s+instructions?\s*[:\-]"),
        re.compile(r"(?i)system\s*[:\-]\s*you\s+are\s+now"),
        re.compile(r"(?i)act\s+as\s+(?:if\s+you\s+are\s+)?(?:a|an)\s+"),
        re.compile(r"(?i)pretend\s+to\s+be\s+(?:a|an)\s+"),
        re.compile(r"(?i)roleplay\s+(?:as\s+)?(?:a|an)\s+"),
    ]

    REVEAL_PROMPT_PATTERNS = [
        re.compile(r"(?i)(?:show|reveal|print|display|output)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)"),
        re.compile(r"(?i)what\s+(?:is\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)"),
        re.compile(r"(?i)repeat\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)"),
        re.compile(r"(?i)tell\s+me\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)"),
    ]

    BYPASS_SAFETY_PATTERNS = [
        re.compile(r"(?i)bypass\s+(?:safety|security|guardrails?|filters?)"),
        re.compile(r"(?i)disable\s+(?:safety|security|guardrails?|filters?)"),
        re.compile(r"(?i)turn\s+off\s+(?:safety|security|guardrails?|filters?)"),
        re.compile(r"(?i)ignore\s+(?:safety|security|guardrails?|filters?)"),
        re.compile(r"(?i)jailbreak"),
        re.compile(r"(?i)do\s+anything\s+now"),
        re.compile(r"(?i)dan\s+mode"),
        re.compile(r"(?i)unrestricted\s+mode"),
    ]

    REDIRECT_TASK_PATTERNS = [
        re.compile(r"(?i)instead\s+of\s+(?:answering|responding|doing)\s+(?:this|that)"),
        re.compile(r"(?i)don't\s+(?:answer|respond|do)\s+(?:this|that)"),
        re.compile(r"(?i)skip\s+(?:this|the\s+(?:question|task|prompt))"),
        re.compile(r"(?i)answer\s+(?:a\s+)?different\s+(?:question|prompt)"),
    ]

    EXFILTRATION_PATTERNS = [
        re.compile(r"(?i)(?:send|transmit|exfiltrate|leak)\s+(?:data|information|secrets?|passwords?|tokens?|keys?)"),
        re.compile(r"(?i)upload\s+(?:to|file)"),
        re.compile(r"(?i)call\s+(?:home|external|api)"),
        re.compile(r"(?i)make\s+(?:a\s+)?(?:request|call)\s+to"),
    ]

    CONFLICTING_INSTRUCTION_PATTERNS = [
        re.compile(r"(?i)(?:if|when)\s+.+\s+(?:then|do)\s+.+\s+(?:but|however)\s+.+\s+(?:don't|do\s+not)"),
        re.compile(r"(?i)actually\s+,?\s+(?:do|follow)\s+.+\s+instead"),
        re.compile(r"(?i)correction\s*[:\-]\s*.+\s+is\s+(?:wrong|incorrect)"),
    ]

    CONTEXT_WEIGHT = 3
    MAX_CONTEXT_WINDOW = 200

    def detect(self, context: DetectionContext) -> list[Finding]:
        findings = []

        if not context.records:
            return findings

        for row_idx, record in enumerate(context.records):
            for col_name, value in record.items():
                if not isinstance(value, str) or len(value) < 10:
                    continue

                findings.extend(self._detect_override_instructions(context, row_idx, col_name, value))
                findings.extend(self._detect_reveal_prompt(context, row_idx, col_name, value))
                findings.extend(self._detect_bypass_safety(context, row_idx, col_name, value))
                findings.extend(self._detect_redirect_task(context, row_idx, col_name, value))
                findings.extend(self._detect_exfiltration(context, row_idx, col_name, value))
                findings.extend(self._detect_conflicting_instructions(context, row_idx, col_name, value))

        return findings

    def _detect_override_instructions(self, context: DetectionContext, row_idx: int, col_name: str, value: str) -> list[Finding]:
        findings = []
        for pattern in self.OVERRIDE_INSTRUCTION_PATTERNS:
            matches = pattern.finditer(value)
            for match in matches:
                finding = self._create_finding(
                    rule_id=PromptInjectionRuleType.OVERRIDE_INSTRUCTION.value,
                    severity=FindingSeverity.HIGH,
                    confidence=FindingConfidence.HIGH,
                    title="Instruction override attempt detected",
                    description=f"Text attempts to override previous instructions in column '{col_name}'",
                    row_number=row_idx + 1,
                    column_name=col_name,
                    evidence=match.group(),
                    masked_evidence=match.group(),
                    remediation_guidance="Text attempts to manipulate model behavior by overriding instructions. Review and sanitize.",
                    metadata={"pattern_type": "override_instruction", "match": match.group()},
                )
                findings.append(finding)
        return findings

    def _detect_reveal_prompt(self, context: DetectionContext, row_idx: int, col_name: str, value: str) -> list[Finding]:
        findings = []
        for pattern in self.REVEAL_PROMPT_PATTERNS:
            matches = pattern.finditer(value)
            for match in matches:
                finding = self._create_finding(
                    rule_id=PromptInjectionRuleType.REVEAL_PROMPT.value,
                    severity=FindingSeverity.HIGH,
                    confidence=FindingConfidence.HIGH,
                    title="System prompt reveal attempt",
                    description=f"Text attempts to reveal system prompt in column '{col_name}'",
                    row_number=row_idx + 1,
                    column_name=col_name,
                    evidence=match.group(),
                    masked_evidence=match.group(),
                    remediation_guidance="Text attempts to extract system prompts. This is a security concern.",
                    metadata={"pattern_type": "reveal_prompt", "match": match.group()},
                )
                findings.append(finding)
        return findings

    def _detect_bypass_safety(self, context: DetectionContext, row_idx: int, col_name: str, value: str) -> list[Finding]:
        findings = []
        for pattern in self.BYPASS_SAFETY_PATTERNS:
            matches = pattern.finditer(value)
            for match in matches:
                finding = self._create_finding(
                    rule_id=PromptInjectionRuleType.BYPASS_SAFETY.value,
                    severity=FindingSeverity.CRITICAL,
                    confidence=FindingConfidence.HIGH,
                    title="Safety bypass attempt",
                    description=f"Text attempts to bypass safety controls in column '{col_name}'",
                    row_number=row_idx + 1,
                    column_name=col_name,
                    evidence=match.group(),
                    masked_evidence=match.group(),
                    remediation_guidance="Text attempts to disable safety mechanisms. This is a severe security concern.",
                    metadata={"pattern_type": "bypass_safety", "match": match.group()},
                )
                findings.append(finding)
        return findings

    def _detect_redirect_task(self, context: DetectionContext, row_idx: int, col_name: str, value: str) -> list[Finding]:
        findings = []
        for pattern in self.REDIRECT_TASK_PATTERNS:
            matches = pattern.finditer(value)
            for match in matches:
                finding = self._create_finding(
                    rule_id=PromptInjectionRuleType.REDIRECT_TASK.value,
                    severity=FindingSeverity.MEDIUM,
                    confidence=FindingConfidence.MEDIUM,
                    title="Task redirection attempt",
                    description=f"Text attempts to redirect model to different task in column '{col_name}'",
                    row_number=row_idx + 1,
                    column_name=col_name,
                    evidence=match.group(),
                    masked_evidence=match.group(),
                    remediation_guidance="Text attempts to divert model from intended task. Review context.",
                    metadata={"pattern_type": "redirect_task", "match": match.group()},
                )
                findings.append(finding)
        return findings

    def _detect_exfiltration(self, context: DetectionContext, row_idx: int, col_name: str, value: str) -> list[Finding]:
        findings = []
        for pattern in self.EXFILTRATION_PATTERNS:
            matches = pattern.finditer(value)
            for match in matches:
                finding = self._create_finding(
                    rule_id=PromptInjectionRuleType.TOOL_EXFILTRATION.value,
                    severity=FindingSeverity.HIGH,
                    confidence=FindingConfidence.MEDIUM,
                    title="Potential data exfiltration instruction",
                    description=f"Text contains potential data exfiltration instruction in column '{col_name}'",
                    row_number=row_idx + 1,
                    column_name=col_name,
                    evidence=match.group(),
                    masked_evidence=match.group(),
                    remediation_guidance="Text contains instructions that could lead to data exfiltration. Review carefully.",
                    metadata={"pattern_type": "exfiltration", "match": match.group()},
                )
                findings.append(finding)
        return findings

    def _detect_conflicting_instructions(self, context: DetectionContext, row_idx: int, col_name: str, value: str) -> list[Finding]:
        findings = []
        for pattern in self.CONFLICTING_INSTRUCTION_PATTERNS:
            matches = pattern.finditer(value)
            for match in matches:
                finding = self._create_finding(
                    rule_id=PromptInjectionRuleType.CONFLICTING_INSTRUCTION.value,
                    severity=FindingSeverity.MEDIUM,
                    confidence=FindingConfidence.LOW,
                    title="Conflicting instructions detected",
                    description=f"Text contains potentially conflicting instructions in column '{col_name}'",
                    row_number=row_idx + 1,
                    column_name=col_name,
                    evidence=match.group(),
                    masked_evidence=match.group(),
                    remediation_guidance="Text contains contradictory instructions. May indicate manipulation attempt.",
                    metadata={"pattern_type": "conflicting_instruction", "match": match.group()},
                )
                findings.append(finding)
        return findings