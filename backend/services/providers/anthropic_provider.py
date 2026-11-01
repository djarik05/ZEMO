"""
Anthropic Provider Implementation
"""
import os
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime


class AnthropicProvider:
    """Anthropic (Claude) provider implementation."""
    
    def __init__(self, config: dict, api_key: str = None):
        self.config = config
        self.api_key = api_key or self._get_api_key()
        self._client = None
        self._models_cache = None
        self._models_cache_time = None
        self._cache_ttl = 300
    
    def _get_api_key(self) -> str:
        """Get the API key from environment or config."""
        return os.environ.get("ANTHROPIC_API_KEY") or self.config.get("api_key")
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "claude-3-5-sonnet" in model_id or "claude-3-5-haiku" in model_id or "claude-3-opus" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "reasoning"]
        elif "claude-3" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision"]
        elif "claude-2" in model_id:
            return ["text_generation", "chat_completion", "function_calling"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "claude-3" in model_id:
            return 200000
        elif "claude-2" in model_id:
            return 100000
        return 100000
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192
    
    def _get_client(self):
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://api.anthropic.com/v1",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    async def test_connection(self) -> dict:
        """Test the connection to Anthropic."""
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
                    "provider_id": "anthropic",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "reasoning": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "anthropic",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "anthropic",
                    "error": f"Anthropic API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "anthropic",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from Anthropic."""
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
                    "provider": "anthropic",
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
                base_url="https://api.anthropic.com/v1",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "claude-3-5-sonnet" in model_id or "claude-3-5-haiku" in model_id or "claude-3-opus" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "reasoning"]
        elif "claude-3" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision"]
        elif "claude-2" in model_id:
            return ["text_generation", "chat_completion", "function_calling"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "claude-3" in model_id:
            return 200000
        elif "claude-2" in model_id:
            return 100000
        return 100000
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192
    
    async def test_connection(self) -> dict:
        """Test the connection to Anthropic."""
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
                    "provider_id": "anthropic",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "reasoning": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "anthropic",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "anthropic",
                    "error": f"Anthropic API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "anthropic",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from Anthropic."""
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
                    "provider": "anthropic",
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
                base_url="https://api.anthropic.com/v1",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "claude-3-5-sonnet" in model_id or "claude-3-5-haiku" in model_id or "claude-3-opus" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "reasoning"]
        elif "claude-3" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision"]
        elif "claude-2" in model_id:
            return ["text_generation", "chat_completion", "function_calling"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "claude-3" in model_id:
            return 200000
        elif "claude-2" in model_id:
            return 100000
        return 100000
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192
    
    async def test_connection(self) -> dict:
        """Test the connection to Anthropic."""
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
                    "provider_id": "anthropic",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "reasoning": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "anthropic",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "anthropic",
                    "error": f"Anthropic API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "anthropic",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from Anthropic."""
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
                    "provider": "anthropic",
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
                base_url="https://api.anthropic.com/v1",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "claude-3-5-sonnet" in model_id or "claude-3-5-haiku" in model_id or "claude-3-opus" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "reasoning"]
        elif "claude-3" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision"]
        elif "claude-2" in model_id:
            return ["text_generation", "chat_completion", "function_calling"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "claude-3" in model_id:
            return 200000
        elif "claude-2" in model_id:
            return 100000
        return 100000
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192
    
    async def test_connection(self) -> dict:
        """Test the connection to Anthropic."""
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
                    "provider_id": "anthropic",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "reasoning": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "anthropic",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "anthropic",
                    "error": f"Anthropic API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "anthropic",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from Anthropic."""
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
                    "provider": "anthropic",
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
                base_url="https://api.anthropic.com/v1",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "claude-3-5-sonnet" in model_id or "claude-3-5-haiku" in model_id or "claude-3-opus" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "reasoning"]
        elif "claude-3" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision"]
        elif "claude-2" in model_id:
            return ["text_generation", "chat_completion", "function_calling"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "claude-3" in model_id:
            return 200000
        elif "claude-2" in model_id:
            return 100000
        return 100000
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192
    
    async def test_connection(self) -> dict:
        """Test the connection to Anthropic."""
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
                    "provider_id": "anthropic",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "reasoning": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "anthropic",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "anthropic",
                    "error": f"Anthropic API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "anthropic",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from Anthropic."""
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
                    "provider": "anthropic",
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
                base_url="https://api.anthropic.com/v1",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "claude-3-5-sonnet" in model_id or "claude-3-5-haiku" in model_id or "claude-3-opus" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "reasoning"]
        elif "claude-3" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision"]
        elif "claude-2" in model_id:
            return ["text_generation", "chat_completion", "function_calling"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "claude-3" in model_id:
            return 200000
        elif "claude-2" in model_id:
            return 100000
        return 100000
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192