"""Triton Inference Server backend with TensorRT-LLM."""

import asyncio
import json
from typing import Any

import structlog
from pydantic import BaseModel

from src.inference.prompt import PromptManager
from src.inference.protocol import InferenceBackend
from src.inference.truncation import truncate_content
from src.schema.loader import SchemaConfig

logger = structlog.get_logger()


class TritonBackend(InferenceBackend):
    """
    TensorRT-LLM inference via Triton Inference Server.

    Uses gRPC for communication with Triton.
    Supports multiple Triton instances for load balancing.
    """

    def __init__(
        self,
        triton_urls: list[str],
        model_name: str,
        email_analysis_model: type[BaseModel],
        schema_config: SchemaConfig,
        prompt_manager: PromptManager,
        timeout_seconds: float = 30.0,
        max_input_tokens: int = 4096,
    ) -> None:
        self._triton_urls = triton_urls
        self._model_name = model_name
        self._email_analysis_model = email_analysis_model
        self._schema_config = schema_config
        self._prompt_manager = prompt_manager
        self._timeout = timeout_seconds
        self._max_input_tokens = max_input_tokens

        self._clients: list[Any] = []
        self._current_client_index = 0
        self._request_count = 0
        self._error_count = 0
        self._truncation_count = 0

    async def initialize(self) -> None:
        """Initialize Triton gRPC clients."""
        try:
            import tritonclient.grpc.aio as grpcclient
        except ImportError:
            logger.warning(
                "tritonclient not available, using mock mode",
                hint="Install with: pip install tritonclient[all]",
            )
            self._clients = []
            return

        for url in self._triton_urls:
            try:
                client = grpcclient.InferenceServerClient(url=url)
                self._clients.append(client)
                logger.info("triton_client_connected", url=url)
            except Exception as e:
                logger.error("triton_client_connection_failed", url=url, error=str(e))

        if not self._clients:
            logger.warning("no_triton_clients_available")

    async def shutdown(self) -> None:
        """Close all Triton clients."""
        for client in self._clients:
            try:
                await client.close()
            except Exception as e:
                logger.warning("triton_client_close_error", error=str(e))

        self._clients = []
        logger.info("triton_backend_shutdown")

    async def analyze_batch(self, emails: list[str]) -> list[BaseModel]:
        """
        Analyze a batch of emails using Triton.

        For now, processes sequentially. Triton handles internal batching.
        """
        results: list[BaseModel] = []

        for email in emails:
            try:
                result = await self._analyze_single(email)
                results.append(result)
            except Exception as e:
                self._error_count += 1
                logger.error("inference_error", error=str(e))
                raise

        return results

    async def _analyze_single(self, email_content: str) -> BaseModel:
        """Analyze a single email."""
        self._request_count += 1

        # Truncate if needed
        content, was_truncated = truncate_content(
            content=email_content,
            max_tokens=self._max_input_tokens,
        )
        if was_truncated:
            self._truncation_count += 1

        # Render prompt
        prompt = self._prompt_manager.render_analyze_prompt(
            email_content=content,
            schema_config=self._schema_config,
        )

        # Get JSON schema for constrained decoding
        json_schema = self._email_analysis_model.model_json_schema()

        # If no Triton clients, use mock response for development
        if not self._clients:
            return await self._mock_inference(prompt, json_schema)

        # Get next client (round-robin)
        client = self._get_next_client()

        # Call Triton
        response_text = await self._call_triton(client, prompt, json_schema)

        # Parse and validate response
        response_data = json.loads(response_text)
        return self._email_analysis_model.model_validate(response_data)

    async def _call_triton(
        self,
        client: Any,
        prompt: str,
        json_schema: dict[str, Any],
    ) -> str:
        """
        Call Triton inference server.

        Note: Actual implementation depends on your Triton model configuration.
        This is a template that should be adapted to your setup.
        """
        import tritonclient.grpc.aio as grpcclient
        import numpy as np

        # Prepare inputs - adjust based on your model's expected inputs
        inputs = []

        # Text input
        text_input = grpcclient.InferInput("text_input", [1], "BYTES")
        text_input.set_data_from_numpy(np.array([prompt.encode()], dtype=object))
        inputs.append(text_input)

        # JSON schema for constrained decoding (if supported by your model)
        schema_input = grpcclient.InferInput("json_schema", [1], "BYTES")
        schema_input.set_data_from_numpy(
            np.array([json.dumps(json_schema).encode()], dtype=object)
        )
        inputs.append(schema_input)

        # Request outputs
        outputs = [grpcclient.InferRequestedOutput("text_output")]

        # Call inference
        response = await asyncio.wait_for(
            client.infer(
                model_name=self._model_name,
                inputs=inputs,
                outputs=outputs,
            ),
            timeout=self._timeout,
        )

        # Extract response text
        output_data = response.as_numpy("text_output")
        return output_data[0].decode()

    async def _mock_inference(
        self,
        prompt: str,  # noqa: ARG002
        json_schema: dict[str, Any],  # noqa: ARG002
    ) -> BaseModel:
        """
        Mock inference for development without Triton.

        Returns a valid response based on the schema.
        """
        logger.debug("using_mock_inference")

        # Build mock response matching schema
        mock_data: dict[str, Any] = {}

        for field_name, field_config in self._schema_config.fields.items():
            if field_config.type == "string":
                mock_data[field_name] = f"Mock {field_name}"
            elif field_config.type == "enum" and field_config.enum_values:
                mock_data[field_name] = field_config.enum_values[0]
            elif field_config.type == "array":
                if field_config.item_type == "enum" and field_config.enum_values:
                    mock_data[field_name] = [field_config.enum_values[0]]
                else:
                    mock_data[field_name] = ["mock", "keywords", "here"]

        # Simulate some latency
        await asyncio.sleep(0.05)

        return self._email_analysis_model.model_validate(mock_data)

    def _get_next_client(self) -> Any:
        """Get next client using round-robin."""
        if not self._clients:
            raise RuntimeError("No Triton clients available")

        client = self._clients[self._current_client_index]
        self._current_client_index = (self._current_client_index + 1) % len(self._clients)
        return client

    async def health_check(self) -> bool:
        """Check if at least one Triton server is healthy."""
        if not self._clients:
            return False

        for client in self._clients:
            try:
                is_live = await client.is_server_live()
                if is_live:
                    return True
            except Exception:
                continue

        return False

    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        return {
            "backend": "triton",
            "triton_urls": self._triton_urls,
            "connected_clients": len(self._clients),
            "model_name": self._model_name,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "truncation_count": self._truncation_count,
        }
