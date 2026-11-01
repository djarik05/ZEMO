"""
Conversation Manager for Zemo AI Chatbot.
Maintains in-memory multi-turn conversational history with dataset associations and clean session handling.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatTurn:
    role: str  # "user" or "model"
    text: str
    timestamp: float = field(default_factory=time.time)

    def to_gemini_dict(self) -> Dict[str, Any]:
        """Convert to Gemini API content format."""
        return {
            "role": self.role,
            "parts": [{"text": self.text}],
        }


@dataclass
class ConversationSession:
    id: str
    dataset_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turns: List[ChatTurn] = field(default_factory=list)

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append(ChatTurn(role=role, text=text))
        self.updated_at = time.time()
        # Cap conversation length to last 20 messages (10 rounds) to conserve token budget
        if len(self.turns) > 20:
            self.turns = self.turns[-20:]

    def clear(self) -> None:
        self.turns.clear()
        self.updated_at = time.time()

    def get_gemini_contents(self) -> List[Dict[str, Any]]:
        return [t.to_gemini_dict() for t in self.turns]


class ConversationManager:
    """Manages active conversation sessions in memory."""

    def __init__(self, ttl_seconds: int = 86400):
        self.sessions: Dict[str, ConversationSession] = {}
        self.ttl_seconds = ttl_seconds

    def get_or_create(self, conversation_id: Optional[str] = None, dataset_id: Optional[str] = None) -> ConversationSession:
        self._cleanup_expired()

        if conversation_id and conversation_id in self.sessions:
            session = self.sessions[conversation_id]
            if dataset_id and session.dataset_id != dataset_id:
                session.dataset_id = dataset_id
            return session

        new_id = conversation_id or str(uuid.uuid4())
        session = ConversationSession(id=new_id, dataset_id=dataset_id)
        self.sessions[new_id] = session
        return session

    def clear_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.sessions:
            self.sessions[conversation_id].clear()
            return True
        return False

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [cid for cid, s in self.sessions.items() if now - s.updated_at > self.ttl_seconds]
        for cid in expired:
            del self.sessions[cid]


_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
