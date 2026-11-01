"""
Gemini API Client for Zemo AI Chatbot.
Handles secure communication with Google Generative Language API via HTTPX.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import httpx

from core.config import get_settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Async client communicating with Google Gemini API."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model or "gemini-2.5-flash"
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """Check if an API key is available."""
        return bool(self.api_key and self.api_key.strip())

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=45.0)
        return self._http_client

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def generate_response(
        self,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1500,
    ) -> str:
        """
        Send a multi-turn conversation or prompt to Gemini and return the text reply.
        """
        if not self.is_configured:
            logger.warning("Gemini API key is not configured in environment.")
            return (
                "👋 Hello! I am Zemo AI, your machine learning mentor. "
                "The backend Gemini API key is currently not configured or is missing from `.env`. "
                "Please ensure `GEMINI_API_KEY` is provided in your `.env` file to enable live AI responses."
            )

        client = await self._get_client()

        # Build payload matching Gemini v1beta REST API schema
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": 0.95,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": system_instruction}],
            }

        endpoint = f"{self.BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"

        try:
            response = await client.post(endpoint, json=payload)

            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "").strip()
                        if text:
                            return text
                return "I received an empty response from Gemini. Could you rephrase your question?"

            elif response.status_code == 429:
                logger.warning("Gemini API rate limit exceeded: %s", response.text)
                return (
                    "⏳ Gemini API rate limit or quota exceeded momentarily. "
                    "Please wait a few seconds and try asking again."
                )

            elif response.status_code in (401, 403):
                logger.error("Gemini API authentication failed (%d): %s", response.status_code, response.text)
                return (
                    "⚠️ Authentication error: The Gemini API key provided in `.env` was rejected by Google. "
                    "Please verify your `GEMINI_API_KEY`."
                )

            elif response.status_code == 404:
                # If specific model is not found, fallback to gemini-2.0-flash or gemini-flash-latest
                if self.model != "gemini-2.5-flash":
                    logger.info("Model %s returned 404, attempting fallback to gemini-2.5-flash", self.model)
                    self.model = "gemini-2.5-flash"
                    return await self.generate_response(contents, system_instruction, temperature, max_output_tokens)
                return f"⚠️ Model '{self.model}' is currently unavailable on Gemini API."

            else:
                logger.error("Gemini API returned status %d: %s", response.status_code, response.text[:300])
                return (
                    f"I encountered a temporary service issue communicating with Gemini ({response.status_code}). "
                    "Please try again in a moment."
                )

        except httpx.TimeoutException:
            logger.error("Gemini API request timed out.")
            return "⏳ The request to Gemini timed out. Please try asking again."
        except httpx.RequestError as e:
            logger.error("Network error communicating with Gemini API: %s", e)
            return f"🔌 Network connection issue reaching Gemini API: {e}. Please check your internet connectivity."
        except Exception as e:
            logger.error("Unexpected error in GeminiClient: %s", e)
            return f"An unexpected error occurred while generating a response: {e}"


_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
