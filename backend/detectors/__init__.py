from detectors.base import BaseDetector, DetectionContext, Finding
from detectors.privacy import PrivacyDetector
from detectors.secrets import SecretsDetector
from detectors.prompt_injection import PromptInjectionDetector
from detectors.data_quality import DataQualityDetector

__all__ = [
    "BaseDetector",
    "DetectionContext",
    "Finding",
    "PrivacyDetector",
    "SecretsDetector",
    "PromptInjectionDetector",
    "DataQualityDetector",
]

DETECTORS = {
    "privacy": PrivacyDetector(),
    "secrets": SecretsDetector(),
    "prompt_injection": PromptInjectionDetector(),
    "data_quality": DataQualityDetector(),
}

DETECTOR_NAMES = {
    "privacy": "privacy",
    "secrets": "secrets",
    "prompt_injection": "prompt_injection",
    "data_quality": "data_quality",
}

DEFAULT_DETECTORS = ["privacy", "secrets", "prompt_injection", "data_quality"]