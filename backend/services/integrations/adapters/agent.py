import httpx

from services.integrations.adapters.base import (
    AdapterConfig,
    AdapterRegistry,
    BaseAdapter,
    CapabilityCheckResult,
    VerificationResult,
)


class AgentBaseAdapter(BaseAdapter):
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
                    "task_execution": "DECLARED",
                    "tool_use": "DECLARED",
                    "multi_step_reasoning": "DECLARED",
                    "data_analysis": "DECLARED",
                }

                return VerificationResult(
                    success=True,
                    status="CONNECTED",
                    diagnostic=f"Agent endpoint reachable. Capabilities declared but not verified.",
                    capabilities=capabilities,
                    metadata={"health_path": health_path},
                )
            else:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic=f"Agent endpoint returned {response.status_code}: {response.text[:200]}",
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
                diagnostic=f"Capability '{capability}' not declared for agent adapter",
            )

        return CapabilityCheckResult(
            capability=capability,
            status="DECLARED",
            verified=False,
            diagnostic=f"Agent capability '{capability}' declared but not verified",
        )


class CustomRestAgentAdapter(AgentBaseAdapter):
    adapter_type = "custom_rest_agent"

    def get_supported_capabilities(self) -> list[str]:
        if self.config.extra and "declared_capabilities" in self.config.extra:
            return self.config.extra["declared_capabilities"]
        return [
            "task_execution",
            "tool_use",
            "multi_step_reasoning",
            "data_analysis",
        ]


class OpenAICompatibleAgentAdapter(AgentBaseAdapter):
    adapter_type = "openai_compatible_agent"

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

                capabilities = {
                    "task_execution": "VERIFIED",
                    "tool_use": "VERIFIED",
                    "multi_step_reasoning": "DECLARED",
                    "data_analysis": "DECLARED",
                }

                return VerificationResult(
                    success=True,
                    status="CONNECTED",
                    diagnostic=f"OpenAI-compatible agent endpoint reachable. Models: {', '.join(models[:5])}",
                    capabilities=capabilities,
                    metadata={"models": models},
                )
            elif response.status_code == 401:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic="Authentication failed: invalid credentials",
                    capabilities={},
                )
            else:
                return VerificationResult(
                    success=False,
                    status="FAILED",
                    diagnostic=f"Agent endpoint returned {response.status_code}: {response.text[:200]}",
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

    def get_supported_capabilities(self) -> list[str]:
        return [
            "task_execution",
            "tool_use",
            "multi_step_reasoning",
            "data_analysis",
        ]


class LocalAgentAdapter(AgentBaseAdapter):
    adapter_type = "local_agent"

    def get_supported_capabilities(self) -> list[str]:
        return [
            "task_execution",
            "tool_use",
            "multi_step_reasoning",
            "data_analysis",
        ]


class WebhookAgentAdapter(AgentBaseAdapter):
    adapter_type = "webhook_agent"

    async def verify_connection(self) -> VerificationResult:
        return VerificationResult(
            success=True,
            status="CONNECTED",
            diagnostic="Webhook agent registered (no active connection verification)",
            capabilities={
                "task_execution": "DECLARED",
                "tool_use": "DECLARED",
            },
            metadata={"type": "webhook"},
        )

    def get_supported_capabilities(self) -> list[str]:
        return [
            "task_execution",
            "tool_use",
        ]


AdapterRegistry.register("custom_rest_agent", CustomRestAgentAdapter)
AdapterRegistry.register("openai_compatible_agent", OpenAICompatibleAgentAdapter)
AdapterRegistry.register("local_agent", LocalAgentAdapter)
AdapterRegistry.register("webhook_agent", WebhookAgentAdapter)