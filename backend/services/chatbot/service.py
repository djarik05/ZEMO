"""
Zemo AI Chatbot Service.
Orchestrates Gemini Client, Dataset Context Builder, and Conversation Sessions.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from services.chatbot.context_builder import DatasetContextBuilder, get_dataset_context_builder
from services.chatbot.conversation_manager import ConversationManager, get_conversation_manager
from services.chatbot.gemini_client import GeminiClient, get_gemini_client
from services.chatbot.prompts import ZEMO_AI_SYSTEM_PROMPT, format_dataset_context

logger = logging.getLogger(__name__)


class ChatbotService:
    """Core domain service powering Zemo AI Assistant."""

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        context_builder: Optional[DatasetContextBuilder] = None,
        conversation_manager: Optional[ConversationManager] = None,
    ):
        self.client = gemini_client or get_gemini_client()
        self.context_builder = context_builder or get_dataset_context_builder()
        self.conv_manager = conversation_manager or get_conversation_manager()

    async def process_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> Tuple[str, str, bool]:
        """
        Process user message through Gemini with grounded dataset context.
        Returns: (assistant_reply, conversation_id, dataset_context_used)
        """
        clean_msg = message.strip()
        if not clean_msg:
            return "Please type a message or question, and I'll be glad to help you!", conversation_id or "new", False

        # 1. Get or create conversation session
        session = self.conv_manager.get_or_create(conversation_id, dataset_id=dataset_id)
        effective_dataset_id = dataset_id or session.dataset_id

        # 2. Build dataset context if dataset is active
        dataset_summary = ""
        context_used = False
        if effective_dataset_id:
            try:
                dataset_summary = await self.context_builder.build_context_summary(effective_dataset_id)
                if dataset_summary:
                    context_used = True
            except Exception as e:
                logger.warning("Error fetching dataset summary for chatbot: %s", e)

        # 3. Assemble dynamic system prompt with grounded context
        system_instruction = ZEMO_AI_SYSTEM_PROMPT + format_dataset_context(dataset_summary)

        # 4. Record user turn
        session.add_turn(role="user", text=clean_msg)

        # 5. Generate reply from Gemini
        gemini_contents = session.get_gemini_contents()
        reply = await self.client.generate_response(
            contents=gemini_contents,
            system_instruction=system_instruction,
        )

        # 6. Record assistant turn
        session.add_turn(role="model", text=reply)

        return reply, session.id, context_used

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history."""
        return self.conv_manager.clear_conversation(conversation_id)


_chatbot_service: Optional[ChatbotService] = None


def get_chatbot_service() -> ChatbotService:
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service
