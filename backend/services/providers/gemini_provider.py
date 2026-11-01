"""
Google Gemini Provider Implementation
"""
import os
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime


class GoogleGeminiProvider:
    """Google Gemini provider implementation."""
    
    def __init__(self, config: dict, api_key: str = None):
        self.config = config
        self.api_key = api_key or self._get_api_key()
        self._client = None
        self._models_cache = None
        self._models_cache_time = None
        self._cache_ttl = 300
    
    def _get_api_key(self) -> str:
        """Get the API key from environment or config."""
        return os.environ.get("GOOGLE_API_KEY") or self.config.get("api_key")
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "gemini-1.5-pro" in model_id or "gemini-1.5-flash" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini-1.0-pro" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "gemini-1.5" in model_id:
            return 1000000
        elif "gemini-1.0" in model_id:
            return 32768
        return 32768
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192
    
    def _get_client(self):
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://generativelanguage.googleapis.com/v1beta",
                params={"key": self.api_key},
                headers={
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    async def test_connection(self) -> dict:
        """Test the connection to Google Gemini."""
        start = datetime.utcnow()
        try:
            client = await self._get_client()
            response = await self._client.get("/models")
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "").split("/")[-1] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
                
                return {
                    "success": True,
                    "provider_id": "google_gemini",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "embeddings": True,
                        "reasoning": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "google_gemini",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "google_gemini",
                    "error": f"Google Gemini API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "google_gemini",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from Google Gemini."""
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
            for model in data.get("models", []):
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    model_id = model.get("name", "").split("/")[-1]
                    capabilities = self._get_model_capabilities(model_id)
                    models.append({
                        "id": model_id,
                        "name": model_id,
                        "provider": "google_gemini",
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
                base_url="https://generativelanguage.googleapis.com/v1beta",
                params={"key": self.api_key},
                headers={
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "gemini-1.5-pro" in model_id or "gemini-1.5-flash" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini-1.0-pro" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "gemini-1.5" in model_id:
            return 1000000
        elif "gemini-1.0" in model_id:
            return 32768
        return 32768
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192
    
    async def test_connection(self) -> dict:
        """Test the connection to Google Gemini."""
        start = datetime.utcnow()
        try:
            client = await self._get_client()
            response = await self._client.get("/models")
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "").split("/")[-1] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
                
                return {
                    "success": True,
                    "provider_id": "google_gemini",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "embeddings": True,
                        "reasoning": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "google_gemini",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "google_gemini",
                    "error": f"Google Gemini API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "google_gemini",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from Google Gemini."""
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
            for model in data.get("models", []):
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    model_id = model.get("name", "").split("/")[-1]
                    capabilities = self._get_model_capabilities(model_id)
                    models.append({
                        "id": model_id,
                        "name": model_id,
                        "provider": "google_gemini",
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
                base_url="https://generativelanguage.googleapis.com/v1beta",
                params={"key": self.api_key},
                headers={
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "gemini-1.5-pro" in model_id or "gemini-1.5-flash" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini-1.0-pro" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "gemini-1.5" in model_id:
            return 1000000
        elif "gemini-1.0" in model_id:
            return 32768
        return 32768
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192
    
    async def test_connection(self) -> dict:
        """Test the connection to Google Gemini."""
        start = datetime.utcnow()
        try:
            client = await self._get_client()
            response = await self._client.get("/models")
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "").split("/")[-1] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
                
                return {
                    "success": True,
                    "provider_id": "google_gemini",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "embeddings": True,
                        "reasoning": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "google_gemini",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "google_gemini",
                    "error": f"Google Gemini API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "google_gemini",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from Google Gemini."""
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
            for model in data.get("models", []):
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    model_id = model.get("name", "").split("/")[-1]
                    capabilities = self._get_model_capabilities(model_id)
                    models.append({
                        "id": model_id,
                        "name": model_id,
                        "provider": "google_gemini",
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
                base_url="https://generativelanguage.googleapis.com/v1beta",
                params={"key": self.api_key},
                headers={
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "gemini-1.5-pro" in model_id or "gemini-1.5-flash" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini-1.0-pro" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "gemini-1.5" in model_id:
            return 1000000
        elif "gemini-1.0" in model_id:
            return 32768
        return 32768
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192
    
    async def test_connection(self) -> dict:
        """Test the connection to Google Gemini."""
        start = datetime.utcnow()
        try:
            client = await self._get_client()
            response = await self._client.get("/models")
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "").split("/")[-1] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
                
                return {
                    "success": True,
                    "provider_id": "google_gemini",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "embeddings": True,
                        "reasoning": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "google_gemini",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "google_gemini",
                    "error": f"Google Gemini API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "google_gemini",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from Google Gemini."""
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
            for model in data.get("models", []):
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    model_id = model.get("name", "").split("/")[-1]
                    capabilities = self._get_model_capabilities(model_id)
                    models.append({
                        "id": model_id,
                        "name": model_id,
                        "provider": "google_gemini",
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
                base_url="https://generativelanguage.googleapis.com/v1beta",
                params={"key": self.api_key},
                headers={
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "gemini-1.5-pro" in model_id or "gemini-1.5-flash" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini-1.0-pro" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "gemini-1.5" in model_id:
            return 1000000
        elif "gemini-1.0" in model_id:
            return 32768
        return 32768
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192
    
    async def test_connection(self) -> dict:
        """Test the connection to Google Gemini."""
        start = datetime.utcnow()
        try:
            client = await self._get_client()
            response = await self._client.get("/models")
            
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "").split("/")[-1] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
                
                return {
                    "success": True,
                    "provider_id": "google_gemini",
                    "latency_ms": latency,
                    "models_found": models,
                    "capabilities": {
                        "text_generation": True,
                        "chat_completion": True,
                        "function_calling": True,
                        "vision": True,
                        "embeddings": True,
                        "reasoning": True
                    }
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "provider_id": "google_gemini",
                    "error": "Authentication failed: invalid API key",
                    "models_found": [],
                    "capabilities": {}
                }
            else:
                return {
                    "success": False,
                    "provider_id": "google_gemini",
                    "error": f"Google Gemini API returned {response.status_code}: {response.text[:200]}",
                    "models_found": [],
                    "capabilities": {}
                }
        except Exception as e:
            return {
                "success": False,
                "provider_id": "google_gemini",
                "error": str(e),
                "models_found": [],
                "capabilities": {}
            }
    
    async def list_models(self) -> List[dict]:
        """List available models from Google Gemini."""
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
            for model in data.get("models", []):
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    model_id = model.get("name", "").split("/")[-1]
                    capabilities = self._get_model_capabilities(model_id)
                    models.append({
                        "id": model_id,
                        "name": model_id,
                        "provider": "google_gemini",
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
                base_url="https://generativelanguage.googleapis.com/v1beta",
                params={"key": self.api_key},
                headers={
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        return self._client
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get capabilities for a specific model."""
        if "gemini-1.5-pro" in model_id or "gemini-1.5-flash" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini-1.0-pro" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        elif "gemini" in model_id:
            return ["text_generation", "chat_completion", "function_calling", "vision", "embeddings", "reasoning"]
        return ["text_generation", "chat_completion"]
    
    def _get_context_window(self, model_id: str) -> int:
        if "gemini-1.5" in model_id:
            return 1000000
        elif "gemini-1.0" in model_id:
            return 32768
        return 32768
    
    def _get_max_output_tokens(self, model_id: str) -> int:
        return 8192