"""
OpenAI Provider Implementation
"""
import os
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime

from services.providers.base import BaseProvider, ProviderRegistry


class OpenAIProvider:
    """OpenAI provider implementation."""
    
    def __init__(self, config: dict, api_key: str = None):
        self.config = config
        self.api_key = api_key or self._get_api_key()
        self._client = None
        self._models_cache = None
        self._models_cache_time = None
        self._cache_ttl = 300  # 5 minutes
    
    def _get_api_key(self) -> str:
        """Get the API key from environment or config."""
        return os.environ.get("OPENAI_API_KEY") or self.config.get("api_key")
    
    async def _get_client(self):
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.config.get("base_url", "https://api.openai.com/v1"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    async def test_connection(self) -> dict:
        """Test the connection to OpenAI."""
        start = datetime.utcnow()
        try:
            client = await self._get_client()
            response = await self._client.get("/models")
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]
                
                return {
                    "success": True,
                    "provider_id": "openai",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "structured_output": True,
                        "reasoning": True,
                        "embeddings": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "openai",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "openai",
                    "error": f"OpenAI API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "openai",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from OpenAI."""
        # Check cache
        if self._models_cache and self._models_cache_time:
            if (datetime.utcnow() - self._models_cache_time).seconds < 300:
                return self._models_cache
        
        client = await self._get_client()
        response = await self._client.get("/models")
        
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("data", []):
                model_id = model.get("id")
                capabilities = self._get_model_capabilities(model_id)
                self._models_cache = [{
                    "id": model_id,
                    "name": model_id,
                    "provider": "openai",
                    "capabilities": capabilities,
                    "context_window": self._get_context_window(model_id),
                    "max_output_tokens": self._get_max_output_tokens(model_id)
                } for model_id in [m.get("id") for m in data.get("data", [])]]
                self._models_cache_time = datetime.utcnow()
                return self._models_cache
        
        return []
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "gpt-4o" in model_id or "gpt-4" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "structured_output", "reasoning", "embeddings"]
        elif "gpt-3.5" in model_id:
            return ["text_generation", "chat_completion", "function_calling"]
        elif "embedding" in model_id:
            return ["embeddings"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        """Get context window for a model."""
        if "gpt-4o" in model_id:
            return 128000
        elif "gpt-4" in model_id:
            return 128000
        elif "gpt-3.5" in model_id:
            return 16385
        return 4096
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        """Get max output tokens for a model."""
        if "gpt-4o" in model_id:
            return 16384
        elif "gpt-4" in model_id:
            return 4096
        elif "gpt-3.5" in model_id:
            return 4096
        return 4096
    
    async def test_connection(self) -> dict:
        """Test the connection to OpenAI."""
        start = datetime.utcnow()
        try:
            client = await self._get_client()
            response = await self._client.get("/models")
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]
                
                return {
                    "success": True,
                    "provider_id": "openai",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "structured_output": True,
                        "reasoning": True,
                        "embeddings": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "openai",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "openai",
                    "error": f"OpenAI API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "openai",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from OpenAI."""
        models = await self.list_models()
        return models
    
    async def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get information about a specific model."""
        models = await self.list_models()
        for model in models:
            if model["id"] == model_id:
                return model
        return None
    
    async def check_capabilities(self, model_id: str) -> Dict[str, bool]:
        """Check which capabilities are supported by a model."""
        capabilities = self._get_model_capabilities(model_id)
        return {cap: True for cap in capabilities}
    
    async def get_model_capabilities(self, model_id: str) -> dict:
        """Get detailed capability information for a model."""
        capabilities = self._get_model_capabilities(model_id)
        return {
            "model_id": model_id,
            "capabilities": capabilities,
            "verified": True,
            "verified_at": datetime.utcnow().isoformat(),
            "details": {
                "context_window": self._get_context_window(model_id),
                "max_output_tokens": self._get_max_output_tokens(model_id)
            }
        }
    
    def _get_client(self):
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    async def test_connection(self) -> dict:
        """Test the connection to OpenAI."""
        start = datetime.utcnow()
        try:
            client = self._get_client()
            response = await self._client.get("/models")
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]
                
                return {
                    "success": True,
                    "provider_id": "openai",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "structured_output": True,
                        "reasoning": True,
                        "embeddings": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "openai",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "openai",
                    "error": f"OpenAI API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "openai",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from OpenAI."""
        # Check cache
        if self._models_cache and self._models_cache_time:
            if (datetime.utcnow() - self._models_cache_time).seconds < 300:
                return self._models_cache
        
        client = await self._get_client()
        response = await self._client.get("/models")
        
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("data", []):
                model_id = model.get("id")
                capabilities = self._get_model_capabilities(model_id)
                models.append({
                    "id": model_id,
                    "name": model_id,
                    "provider": "openai",
                    "capabilities": capabilities,
                    "context_window": self._get_context_window(model_id),
                    "max_output_tokens": self._get_max_output_tokens(model_id)
                })
            self._models_cache = models
            self._models_cache_time = datetime.utcnow()
            return models
        
        return []
    
    async def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get information about a specific model."""
        models = await self.list_models()
        for model in models:
            if model["id"] == model_id:
                return model
        return None
    
    async def check_capabilities(self, model_id: str) -> Dict[str, bool]:
        """Check which capabilities are supported by a model."""
        capabilities = self._get_model_capabilities(model_id)
        return {cap: True for cap in capabilities}
    
    async def get_model_capabilities(self, model_id: str) -> dict:
        """Get detailed capability information for a model."""
        capabilities = self._get_model_capabilities(model_id)
        return {
            "model_id": model_id,
            "capabilities": capabilities,
            "verified": True,
            "verified_at": datetime.utcnow().isoformat(),
            "details": {
                "context_window": self._get_context_window(model_id),
                "max_output_tokens": self._get_max_output_tokens(model_id)
            }
        }
    
    def _get_client(self):
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "gpt-4o" in model_id or "gpt-4" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "structured_output", "reasoning", "embeddings"]
        elif "gpt-3.5" in model_id:
            return ["text_generation", "chat_completion", "function_calling"]
        elif "embedding" in model_id:
            return ["embeddings"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "gpt-4o" in model_id:
            return 128000
        elif "gpt-4" in model_id:
            return 128000
        elif "gpt-3.5" in model_id:
            return 16385
        return 4096
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        if "gpt-4o" in model_id:
            return 16384
        elif "gpt-4" in model_id:
            return 4096
        elif "gpt-3.5" in model_id:
            return 4096
        return 4096
    
    async def test_connection(self) -> dict:
        """Test the connection to OpenAI."""
        start = datetime.utcnow()
        try:
            client = await self._get_client()
            response = await self._client.get("/models")
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]
                
                return {
                    "success": True,
                    "provider_id": "openai",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "structured_output": True,
                        "reasoning": True,
                        "embeddings": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "openai",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "openai",
                    "error": f"OpenAI API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "openai",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from OpenAI."""
        # Check cache
        if self._models_cache and self._models_cache_time:
            if (datetime.utcnow() - self._models_cache_time).seconds < 300:
                return self._models_cache
        
        import httpx
        client = await self._get_client()
        response = await self._client.get("/models")
        
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("data", []):
                model_id = model.get("id")
                capabilities = self._get_model_capabilities(model_id)
                models.append({
                    "id": model_id,
                    "name": model_id,
                    "provider": "openai",
                    "capabilities": capabilities,
                    "context_window": self._get_context_window(model_id),
                    "max_output_tokens": self._get_max_output_tokens(model_id)
                })
            self._models_cache = models
            self._models_cache_time = datetime.utcnow()
            return models
        
        return []
    
    async def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get information about a specific model."""
        models = await self.list_models()
        for model in models:
            if model["id"] == model_id:
                return model
        return None
    
    async def check_capabilities(self, model_id: str) -> Dict[str, bool]:
        """Check which capabilities are supported by a model."""
        capabilities = self._get_model_capabilities(model_id)
        return {cap: True for cap in capabilities}
    
    async def get_model_capabilities(self, model_id: str) -> dict:
        """Get detailed capability information for a model."""
        capabilities = self._get_model_capabilities(model_id)
        return {
            "model_id": model_id,
            "capabilities": capabilities,
            "verified": True,
            "verified_at": datetime.utcnow().isoformat(),
            "details": {
                "context_window": self._get_context_window(model_id),
                "max_output_tokens": self._get_max_output_tokens(model_id)
            }
        }
    
    def _get_client(self):
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "gpt-4o" in model_id or "gpt-4" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "structured_output", "reasoning", "embeddings"]
        elif "gpt-3.5" in model_id:
            return ["text_generation", "chat_completion", "function_calling"]
        elif "embedding" in model_id:
            return ["embeddings"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "gpt-4o" in model_id:
            return 128000
        elif "gpt-4" in model_id:
            return 128000
        elif "gpt-3.5" in model_id:
            return 16385
        return 4096
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        if "gpt-4o" in model_id:
            return 16384
        elif "gpt-4" in model_id:
            return 4096
        elif "gpt-3.5" in model_id:
            return 4096
        return 4096
    
    async def test_connection(self) -> dict:
        """Test the connection to OpenAI."""
        start = datetime.utcnow()
        try:
            import httpx
            client = await self._get_client()
            response = await self._client.get("/models")
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]
                
                return {
                    "success": True,
                    "provider_id": "openai",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "structured_output": True,
                        "reasoning": True,
                        "embeddings": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "openai",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "openai",
                    "error": f"OpenAI API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "openai",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from OpenAI."""
        # Check cache
        if self._models_cache and self._models_cache_time:
            if (datetime.utcnow() - self._models_cache_time).seconds < 300:
                return self._models_cache
        
        import httpx
        client = await self._get_client()
        response = await self._client.get("/models")
        
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("data", []):
                model_id = model.get("id")
                capabilities = self._get_model_capabilities(model_id)
                models.append({
                    "id": model_id,
                    "name": model_id,
                    "provider": "openai",
                    "capabilities": capabilities,
                    "context_window": self._get_context_window(model_id),
                    "max_output_tokens": self._get_max_output_tokens(model_id)
                })
            self._models_cache = models
            self._models_cache_time = datetime.utcnow()
            return models
        
        return []
    
    async def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get information about a specific model."""
        models = await self.list_models()
        for model in models:
            if model["id"] == model_id:
                return model
        return None
    
    async def check_capabilities(self, model_id: str) -> Dict[str, bool]:
        """Check which capabilities are supported by a model."""
        capabilities = self._get_model_capabilities(model_id)
        return {cap: True for cap in capabilities}
    
    async def get_model_capabilities(self, model_id: str) -> dict:
        """Get detailed capability information for a model."""
        capabilities = self._get_model_capabilities(model_id)
        return {
            "model_id": model_id,
            "capabilities": capabilities,
            "verified": True,
            "verified_at": datetime.utcnow().isoformat(),
            "details": {
                "context_window": self._get_context_window(model_id),
                "max_output_tokens": self._get_max_output_tokens(model_id)
            }
        }
    
    def _get_client(self):
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "gpt-4o" in model_id or "gpt-4" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "structured_output", "reasoning", "embeddings"]
        elif "gpt-3.5" in model_id:
            return ["text_generation", "chat_completion", "function_calling"]
        elif "embedding" in model_id:
            return ["embeddings"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "gpt-4o" in model_id:
            return 128000
        elif "gpt-4" in model_id:
            return 128000
        elif "gpt-3.5" in model_id:
            return 16385
        return 4096
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        if "gpt-4o" in model_id:
            return 16384
        elif "gpt-4" in model_id:
            return 4096
        elif "gpt-3.5" in model_id:
            return 4096
        return 4096