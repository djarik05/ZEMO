import httpx

from services.integrations.adapters.base import (
    AdapterConfig,
    AdapterRegistry,
    BaseAdapter,
    CapabilityCheckResult,
    VerificationResult,
)


class CustomRestAdapter(BaseAdapter):
    adapter_type = "custom_rest"

    async def verify_connection(self) -> VerificationResult:
        headers = self._build_headers()
        health_path = self.config.extra.get("health_path", "/health") if self.config.extra else "/health"
        url = f"{self.config.base_url.rstrip('/')}{health_path}"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                max_response_size=self.max_response_size,
                follow_redirects=False,
            ) as client:
                response = await client.get(url, headers=headers)

            if response.status_code == 200:
                capabilities = {
                    "text_generation": "DECLARED",
                    "embeddings": "DECLARED",
                    "function_calling": "DECLARED",
                    "training": "DECLARED",
                }

                return VerificationResult(
                    success=True,
                    status="CONNECTED",
                    diagnostic=f"Custom REST endpoint reachable. Capabilities not verified.",
                    capabilities=capabilities,
                    metadata={"health_path": health_path},
                )
            else:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic=f"Custom REST endpoint returned {response.status_code}: {response.text[:200]}",
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
                diagnostic=f"Capability '{capability}' not declared for custom adapter",
            )

        return CapabilityCheckResult(
            capability=capability,
            status="DECLARED",
            verified=False,
            diagnostic=f"Capability '{capability}' declared but not verified (custom adapter)",
        )

    def get_supported_capabilities(self) -> list[str]:
        if self.config.extra and "declared_capabilities" in self.config.extra:
            return self.config.extra["declared_capabilities"]
        return [
            "text_generation",
            "embeddings",
            "function_calling",
            "training",
        ]


AdapterRegistry.register("custom_rest", CustomRestAdapter)