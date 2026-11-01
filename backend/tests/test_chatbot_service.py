"""
Tests for Phase 8: Zemo AI Assistant (Gemini-Powered Chatbot).
Tests general ML tutoring, dataset-aware grounded intelligence,
conversation history, clear session, and security (no credential exposure).
"""

import io
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

from services.chatbot.context_builder import DatasetContextBuilder
from services.chatbot.conversation_manager import ConversationManager
from services.chatbot.gemini_client import GeminiClient
from services.chatbot.service import ChatbotService
from core.database.models.dataset import Dataset, DatasetFormat, DatasetStatus


@pytest.mark.asyncio
async def test_chatbot_status_endpoint(async_client: AsyncClient):
    """Verify chatbot status endpoint returns configuration status and model name without secrets."""
    res = await async_client.get("/api/v1/chatbot/status")
    assert res.status_code == 200
    data = res.json()
    assert "configured" in data
    assert "model" in data
    assert "gemini" in data["model"].lower()
    # Ensure api key is NEVER exposed in the status response
    assert "api_key" not in data
    assert "AQ." not in str(data)


@pytest.mark.asyncio
async def test_general_ml_tutoring_without_dataset(async_client: AsyncClient):
    """Ask an ML question without a dataset and verify friendly educational response."""
    # Mock GeminiClient to ensure deterministic unit testing
    mock_reply = (
        "One-Hot Encoding is a way of converting text categories into numbers so a machine learning model can understand them.\n\n"
        "Imagine your dataset has a column called City with three values: Bhubaneswar, Delhi, and Mumbai.\n"
        "One-Hot Encoding creates three separate columns:\n"
        "- City_Bhubaneswar\n"
        "- City_Delhi\n"
        "- City_Mumbai\n\n"
        "If a row belongs to Delhi, its values become 0, 1, and 0."
    )

    with patch.object(GeminiClient, "generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_reply

        payload = {
            "message": "What is One-Hot Encoding?",
            "conversation_id": None,
            "dataset_id": None,
        }
        res = await async_client.post("/api/v1/chatbot/message", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "One-Hot Encoding" in data["reply"]
        assert "Delhi" in data["reply"]
        assert data["conversation_id"] is not None
        assert data["dataset_context_used"] is False
        assert "AQ." not in res.text  # No credential leak


@pytest.mark.asyncio
async def test_dataset_aware_intelligence(async_client: AsyncClient):
    """Upload a dataset and verify chatbot grounds recommendations in actual columns."""
    sample_csv = "age,income,homeowner,loan_default\n35,50000,yes,no\n45,85000,yes,no\n22,18000,no,yes\n"
    files = {
        "file": ("loan_data.csv", io.BytesIO(sample_csv.encode("utf-8")), "text/csv")
    }
    upload_res = await async_client.post("/api/v1/datasets", files=files)
    assert upload_res.status_code in (201, 409)
    if upload_res.status_code == 201:
        dataset_id = upload_res.json()["id"]
    else:
        err_data = upload_res.json()
        dataset_id = err_data.get("existing_dataset_id") or err_data["detail"]["existing_dataset_id"]

    # Verify context builder generates grounded factual summary
    builder = DatasetContextBuilder()
    context_str = await builder.build_context_summary(dataset_id)
    assert "loan_data" in context_str or "Uploaded Dataset" in context_str
    assert "age" in context_str
    assert "loan_default" in context_str

    mock_reply = "Looking at your `loan_data.csv` dataset, you have 4 columns. A great target column to predict is `loan_default` (Classification)."
    with patch.object(GeminiClient, "generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_reply

        payload = {
            "message": "Which column should I select as my target?",
            "dataset_id": dataset_id,
        }
        res = await async_client.post("/api/v1/chatbot/message", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["dataset_context_used"] is True
        assert "loan_default" in data["reply"]


@pytest.mark.asyncio
async def test_conversational_follow_up_and_clear_history(async_client: AsyncClient):
    from services.chatbot.conversation_manager import get_conversation_manager

    conv_mgr = get_conversation_manager()
    session = conv_mgr.get_or_create("test-conv-001")
    session.add_turn("user", "What is scaling?")
    session.add_turn("model", "Scaling adjusts numbers to a standard scale.")
    session.add_turn("user", "Which one should I use?")
    session.add_turn("model", "StandardScaler is great when data is normally distributed.")

    assert len(session.turns) == 4

    # Clear conversation
    res = await async_client.post("/api/v1/chatbot/clear", json={"conversation_id": "test-conv-001"})
    assert res.status_code == 200
    assert res.json()["success"] is True

    # Verify turns cleared
    assert len(session.turns) == 0


@pytest.mark.asyncio
async def test_empty_message_handling(async_client: AsyncClient):
    """Test response to empty or whitespace message."""
    res = await async_client.post("/api/v1/chatbot/message", json={"message": "   "})
    assert res.status_code == 422  # Pydantic min_length validation error
