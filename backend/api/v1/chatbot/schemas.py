"""
Pydantic Schemas for Zemo AI Chatbot.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User prompt or query.")
    conversation_id: Optional[str] = Field(None, description="Active conversation session ID for multi-turn history.")
    dataset_id: Optional[str] = Field(None, description="Active dataset ID to ground responses in user data.")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Message cannot be empty or whitespace.")
        return s


class ChatMessageResponse(BaseModel):
    reply: str = Field(..., description="Zemo AI response message.")
    conversation_id: str = Field(..., description="Conversation session ID for subsequent turns.")
    dataset_id: Optional[str] = Field(None, description="Active dataset ID associated with query.")
    dataset_context_used: bool = Field(False, description="Whether actual dataset context was injected into the prompt.")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the response.")


class ClearChatRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation session ID to reset.")


class ClearChatResponse(BaseModel):
    conversation_id: str
    success: bool
    message: str


class ChatbotStatusResponse(BaseModel):
    configured: bool = Field(..., description="Whether Gemini API key is configured.")
    model: str = Field(..., description="Gemini model name.")
    provider: str = Field(default="Google Gemini")
