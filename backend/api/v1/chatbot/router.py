"""
FastAPI Router for Zemo AI Chatbot.
"""

from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, status

from api.v1.chatbot.schemas import (
    ChatbotStatusResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ClearChatRequest,
    ClearChatResponse,
)
from core.config import get_settings
from services.chatbot.service import ChatbotService, get_chatbot_service

router = APIRouter(prefix="", tags=["chatbot"])


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Send Message to Zemo AI Chatbot",
    description="Ask machine learning questions, receive dataset-tailored explanations, or get guidance through the Zemo AutoML workflow.",
)
async def send_chat_message(
    payload: ChatMessageRequest,
    service: ChatbotService = Depends(get_chatbot_service),
) -> ChatMessageResponse:
    reply, conv_id, context_used = await service.process_message(
        message=payload.message,
        conversation_id=payload.conversation_id,
        dataset_id=payload.dataset_id,
    )

    return ChatMessageResponse(
        reply=reply,
        conversation_id=conv_id,
        dataset_id=payload.dataset_id,
        dataset_context_used=context_used,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.post(
    "/clear",
    response_model=ClearChatResponse,
    summary="Clear Chat Conversation History",
    description="Resets the multi-turn conversational history for the given conversation ID.",
)
async def clear_chat_history(
    payload: ClearChatRequest,
    service: ChatbotService = Depends(get_chatbot_service),
) -> ClearChatResponse:
    success = service.clear_conversation(payload.conversation_id)
    return ClearChatResponse(
        conversation_id=payload.conversation_id,
        success=success,
        message="Conversation history cleared successfully." if success else "Conversation session not found.",
    )


@router.get(
    "/status",
    response_model=ChatbotStatusResponse,
    summary="Get Chatbot Service Status",
    description="Returns whether Gemini API key is configured and active model name.",
)
async def get_chatbot_status() -> ChatbotStatusResponse:
    settings = get_settings()
    configured = bool(settings.gemini_api_key and settings.gemini_api_key.strip())
    return ChatbotStatusResponse(
        configured=configured,
        model=settings.gemini_model,
        provider="Google Gemini",
    )
