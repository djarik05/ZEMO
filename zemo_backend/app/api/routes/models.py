"""Target AI model provider catalog and configuration API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel

from app.models.enums import IntendedOperation
from app.repositories.dataset_repo import DatasetRepository

router = APIRouter()
repo = DatasetRepository()

PROVIDERS_CATALOG: list[dict[str, Any]] = [
    {
        "id": "openai",
        "name": "OpenAI",
        "category": "Cloud Provider",
        "description": "State-of-the-art foundation models with high reasoning capacity and structured outputs.",
        "defaultModel": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "o1-preview", "o3-mini"],
        "capabilities": ["Text Analysis", "Vision", "Function Calling", "Embeddings", "Fine-Tuning"],
        "envDefault": "Cloud Isolated",
    },
    {
        "id": "anthropic",
        "name": "Anthropic",
        "category": "Cloud Provider",
        "description": "Safety-aligned Claude architecture with Constitutional AI principles and large context.",
        "defaultModel": "claude-3-5-sonnet-20241022",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "capabilities": ["Text Analysis", "Vision", "Tool Use", "Large Context"],
        "envDefault": "Cloud Isolated",
    },
    {
        "id": "gemini",
        "name": "Google Gemini",
        "category": "Cloud Provider",
        "description": "Native multimodal foundation models with deep enterprise security integration.",
        "defaultModel": "gemini-1.5-pro",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-thinking"],
        "capabilities": ["Text Analysis", "Vision", "Audio", "Function Calling", "Embeddings"],
        "envDefault": "Cloud Isolated",
    },
    {
        "id": "meta",
        "name": "Meta Llama",
        "category": "Open Weights",
        "description": "Leading open foundation models for private datacenter deployment and full control.",
        "defaultModel": "llama-3.3-70b-instruct",
        "models": ["llama-3.3-70b-instruct", "llama-3.1-8b-instruct", "llama-guard-3-8b"],
        "capabilities": ["Text Analysis", "Fine-Tuning", "Private Enclave", "Self-Hosted"],
        "envDefault": "Hybrid Enclave",
    },
    {
        "id": "mistral",
        "name": "Mistral AI",
        "category": "European Cloud",
        "description": "High-efficiency open and commercial models with European sovereign data guarantees.",
        "defaultModel": "mistral-large-2411",
        "models": ["mistral-large-2411", "pixtral-large-2411", "mistral-small-2409"],
        "capabilities": ["Text Analysis", "Vision", "Function Calling", "Fine-Tuning"],
        "envDefault": "Cloud Isolated",
    },
    {
        "id": "ollama",
        "name": "Ollama / Local",
        "category": "Local Processing",
        "description": "Direct on-device or on-premises execution ensuring zero data egress outside your perimeter.",
        "defaultModel": "llama3.2:latest",
        "models": ["llama3.2:latest", "qwen2.5-coder:32b", "deepseek-r1:14b", "phi4:latest"],
        "capabilities": ["Air-Gapped", "Zero Egress", "On-Premises", "Private Hardware"],
        "envDefault": "Local Air-Gapped",
    },
]

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "provider": "openai",
    "provider_name": "OpenAI",
    "model": "gpt-4o",
    "environment": "Cloud Isolated",
    "task": "Supervised Fine-Tuning",
    "intended_operation": IntendedOperation.FINE_TUNING.value,
    "privacy_level": "Strict",
}

TASK_TO_OPERATION = {
    "Supervised Fine-Tuning": IntendedOperation.FINE_TUNING.value,
    "RAG Knowledge Indexing": IntendedOperation.TRAINING.value,
    "Model Safety Evaluation": IntendedOperation.EVALUATION.value,
    "Direct Inference": IntendedOperation.INFERENCE.value,
    "fine_tuning": IntendedOperation.FINE_TUNING.value,
    "training": IntendedOperation.TRAINING.value,
    "evaluation": IntendedOperation.EVALUATION.value,
    "inference": IntendedOperation.INFERENCE.value,
}


class ModelConfigRequest(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    environment: str = "Cloud Isolated"
    task: str = "Supervised Fine-Tuning"
    privacy_level: str = "Strict"


@router.get("/models/providers")
async def get_providers() -> dict[str, Any]:
    saved_config = repo.get_model_config() or DEFAULT_MODEL_CONFIG
    return {
        "providers": PROVIDERS_CATALOG,
        "active_config": saved_config,
    }


@router.get("/models/config")
async def get_model_config() -> dict[str, Any]:
    return repo.get_model_config() or DEFAULT_MODEL_CONFIG


@router.post("/models/config", status_code=status.HTTP_200_OK)
async def save_model_config(request: ModelConfigRequest) -> dict[str, Any]:
    provider_obj = next(
        (p for p in PROVIDERS_CATALOG if p["id"] == request.provider),
        PROVIDERS_CATALOG[0],
    )
    intended_op = TASK_TO_OPERATION.get(
        request.task, IntendedOperation.TRAINING.value
    )
    config_data = {
        "provider": provider_obj["id"],
        "provider_name": provider_obj["name"],
        "model": request.model or provider_obj["defaultModel"],
        "environment": request.environment,
        "task": request.task,
        "intended_operation": intended_op,
        "privacy_level": request.privacy_level,
    }
    repo.save_model_config(config_data)
    return {
        "status": "saved",
        "config": config_data,
        "message": f"Target configuration saved for {provider_obj['name']} ({config_data['model']}).",
    }
