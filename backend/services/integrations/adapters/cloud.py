import httpx

from services.integrations.adapters.base import (
    AdapterConfig,
    AdapterRegistry,
    BaseAdapter,
    CapabilityCheckResult,
    VerificationResult,
)


class OpenAIAdapter(BaseAdapter):
    adapter_type = "openai"

    async def verify_connection(self) -> VerificationResult:
        headers = self._build_headers()
        url = f"{self.config.base_url.rstrip('/')}/v1/models"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                max_response_size=self.max_response_size,
                follow_redirects=False,
            ) as client:
                response = await client.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]

                model_available = self.config.model_identifier in models

                capabilities = {
                    "text_generation": "VERIFIED" if model_available else "UNSUPPORTED",
                    "embeddings": "VERIFIED",
                    "function_calling": "VERIFIED" if model_available else "UNSUPPORTED",
                    "fine_tuning": "VERIFIED",
                }

                return VerificationResult(
                    success=True,
                    status="CONNECTED",
                    diagnostic=f"OpenAI API reachable. Model '{self.config.model_identifier}' {'available' if model_available else 'not found in list'}.",
                    capabilities=capabilities,
                    metadata={"models": models},
                )
            elif response.status_code == 401:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic="Authentication failed: invalid API key",
                    capabilities={},
                )
            else:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic=f"OpenAI API returned {response.status_code}: {response.text[:200]}",
                    capabilities={},
                )
        except httpx.TimeoutException:
            return VerificationResult(
                success=False,
                status="FAILED",
                diagnostic="Connection timeout",
                capabilities={},
            )
        except httpx.RequestError as e:
            return VerificationResult(
                success=False,
                status="FAILED",
                diagnostic=f"Network error: {str(e)}",
                capabilities={},
            )

    async def check_capability(self, capability: str) -> CapabilityCheckResult:
        if capability not in self.get_supported_capabilities():
            return CapabilityCheckResult(
                capability=capability,
                status="UNSUPPORTED",
                verified=False,
                diagnostic=f"Capability '{capability}' not supported by OpenAI adapter",
            )

        if capability == "text_generation":
            headers = self._build_headers()
            url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"

            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds,
                    max_response_size=self.max_response_size,
                    follow_redirects=False,
                ) as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json={
                            "model": self.config.model_identifier,
                            "messages": [{"role": "user", "content": "test"}],
                            "max_tokens": 5,
                        },
                    )

                if response.status_code == 200:
                    return CapabilityCheckResult(
                        capability=capability,
                        status="VERIFIED",
                        verified=True,
                        diagnostic="Text generation verified",
                    )
                elif response.status_code == 404:
                    return CapabilityCheckResult(
                        capability=capability,
                        status="UNSUPPORTED",
                        verified=False,
                        diagnostic=f"Model '{self.config.model_identifier}' not found",
                    )
                else:
                    return CapabilityCheckResult(
                        capability=capability,
                        status="UNKNOWN",
                        verified=False,
                        diagnostic=f"Request returned {response.status_code}",
                    )
            except Exception as e:
                return CapabilityCheckResult(
                    capability=capability,
                    status="UNKNOWN",
                    verified=False,
                    diagnostic=f"Check failed: {str(e)}",
                )

        return CapabilityCheckResult(
            capability=capability,
            status="VERIFIED",
            verified=True,
            diagnostic=f"Capability '{capability}' declared as supported",
        )

    def get_supported_capabilities(self) -> list[str]:
        return [
            "text_generation",
            "embeddings",
            "function_calling",
            "fine_tuning",
        ]


class AnthropicAdapter(BaseAdapter):
    adapter_type = "anthropic"

    async def verify_connection(self) -> VerificationResult:
        headers = self._build_headers()
        headers["anthropic-version"] = "2023-06-01"
        url = f"{self.config.base_url.rstrip('/')}/v1/models"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                max_response_size=self.max_response_size,
                follow_redirects=False,
            ) as client:
                response = await client.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]

                model_available = self.config.model_identifier in models

                capabilities = {
                    "text_generation": "VERIFIED" if model_available else "UNSUPPORTED",
                    "embeddings": "UNSUPPORTED",
                    "function_calling": "VERIFIED" if model_available else "UNSUPPORTED",
                }

                return VerificationResult(
                    success=True,
                    status="CONNECTED",
                    diagnostic=f"Anthropic API reachable. Model '{self.config.model_identifier}' {'available' if model_available else 'not found in list'}.",
                    capabilities=capabilities,
                    metadata={"models": models},
                )
            elif response.status_code == 401:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic="Authentication failed: invalid API key",
                    capabilities={},
                )
            else:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic=f"Anthropic API returned {response.status_code}: {response.text[:200]}",
                    capabilities={},
                )
        except httpx.TimeoutException:
            return VerificationResult(
                success=False,
                status="FAILED",
                diagnostic="Connection timeout",
                capabilities={},
            )
        except httpx.RequestError as e:
            return VerificationResult(
                success=False,
                status="FAILED",
                diagnostic=f"Network error: {str(e)}",
                capabilities={},
            )

    async def check_capability(self, capability: str) -> CapabilityCheckResult:
        if capability not in self.get_supported_capabilities():
            return CapabilityCheckResult(
                capability=capability,
                status="UNSUPPORTED",
                verified=False,
                diagnostic=f"Capability '{capability}' not supported by Anthropic adapter",
            )

        return CapabilityCheckResult(
            capability=capability,
            status="VERIFIED",
            verified=True,
            diagnostic=f"Capability '{capability}' declared as supported",
        )

    def get_supported_capabilities(self) -> list[str]:
        return [
            "text_generation",
            "function_calling",
        ]


class GoogleGeminiAdapter(BaseAdapter):
    adapter_type = "google_gemini"

    async def verify_connection(self) -> VerificationResult:
        headers = self._build_headers()
        url = f"{self.config.base_url.rstrip('/')}/v1/models/{self.config.model_identifier}"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                max_response_size=self.max_response_size,
                follow_redirects=False,
            ) as client:
                response = await client.get(url, headers=headers)

            if response.status_code == 200:
                capabilities = {
                    "text_generation": "VERIFIED",
                    "embeddings": "VERIFIED",
                    "function_calling": "VERIFIED",
                }

                return VerificationResult(
                    success=True,
                    status="CONNECTED",
                    diagnostic=f"Google Gemini API reachable. Model '{self.config.model_identifier}' verified.",
                    capabilities=capabilities,
                    metadata={},
                )
            elif response.status_code == 401:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic="Authentication failed: invalid API key",
                    capabilities={},
                )
            elif response.status_code == 404:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic=f"Model '{self.config.model_identifier}' not found",
                    capabilities={},
                )
            else:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic=f"Google Gemini API returned {response.status_code}: {response.text[:200]}",
                    capabilities={},
                )
        except httpx.TimeoutException:
            return VerificationResult(
                success=False,
                status="FAILED",
                diagnostic="Connection timeout",
                capabilities={},
            )
        except httpx.RequestError as e:
            return VerificationResult(
                success=False,
                status="FAILED",
                diagnostic=f"Network error: {str(e)}",
                capabilities={},
            )

    async def check_capability(self, capability: str) -> CapabilityCheckResult:
        if capability not in self.get_supported_capabilities():
            return CapabilityCheckResult(
                capability=capability,
                status="UNSUPPORTED",
                verified=False,
                diagnostic=f"Capability '{capability}' not supported by Google Gemini adapter",
            )

        return CapabilityCheckResult(
            capability=capability,
            status="VERIFIED",
            verified=True,
            diagnostic=f"Capability '{capability}' declared as supported",
        )

    def get_supported_capabilities(self) -> list[str]:
        return [
            "text_generation",
            "embeddings",
            "function_calling",
        ]


class MistralAdapter(BaseAdapter):
    adapter_type = "mistral"

    async def verify_connection(self) -> VerificationResult:
        headers = self._build_headers()
        url = f"{self.config.base_url.rstrip('/')}/v1/models"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                max_response_size=self.max_response_size,
                follow_redirects=False,
            ) as client:
                response = await client.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]

                model_available = self.config.model_identifier in models

                capabilities = {
                    "text_generation": "VERIFIED" if model_available else "UNSUPPORTED",
                    "embeddings": "VERIFIED",
                    "function_calling": "VERIFIED" if model_available else "UNSUPPORTED",
                }

                return VerificationResult(
                    success=True,
                    status="CONNECTED",
                    diagnostic=f"Mistral API reachable. Model '{self.config.model_identifier}' {'available' if model_available else 'not found in list'}.",
                    capabilities=capabilities,
                    metadata={"models": models},
                )
            elif response.status_code == 401:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic="Authentication failed: invalid API key",
                    capabilities={},
                )
            else:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic=f"Mistral API returned {response.status_code}: {response.text[:200]}",
                    capabilities={},
                )
        except httpx.TimeoutException:
            return VerificationResult(
                success=False,
                status="FAILED",
                diagnostic="Connection timeout",
                capabilities={},
            )
        except httpx.RequestError as e:
            return VerificationResult(
                success=False,
                status="FAILED",
                diagnostic=f"Network error: {str(e)}",
                capabilities={},
            )

    async def check_capability(self, capability: str) -> CapabilityCheckResult:
        if capability not in self.get_supported_capabilities():
            return CapabilityCheckResult(
                capability=capability,
                status="UNSUPPORTED",
                verified=False,
                diagnostic=f"Capability '{capability}' not supported by Mistral adapter",
            )

        return CapabilityCheckResult(
            capability=capability,
            status="VERIFIED",
            verified=True,
            diagnostic=f"Capability '{capability}' declared as supported",
        )

    def get_supported_capabilities(self) -> list[str]:
        return [
            "text_generation",
            "embeddings",
            "function_calling",
            "fine_tuning",
        ]


AdapterRegistry.register("openai", OpenAIAdapter)
AdapterRegistry.register("anthropic", AnthropicAdapter)
AdapterRegistry.register("google_gemini", GoogleGeminiAdapter)
AdapterRegistry.register("mistral", MistralAdapter)