"""Abstract protocol for inference backends."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class InferenceBackend(ABC):
    """
    Abstract base for LLM inference backends.

    Implementations:
    - TritonBackend: TensorRT-LLM via Triton Inference Server
    - (Future) vLLMBackend: vLLM server
    - (Future) MockBackend: For testing
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend connection.

        Called once at application startup.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Cleanup backend resources.

        Called once at application shutdown.
        """
        ...

    @abstractmethod
    async def analyze_batch(self, emails: list[str]) -> list[BaseModel]:
        """
        Analyze a batch of emails.

        Args:
            emails: List of email contents (plain text or markdown)

        Returns:
            List of EmailAnalysis model instances, one per email
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the backend is healthy and ready.

        Returns:
            True if backend is ready to accept requests
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get backend statistics.

        Returns:
            Dict with backend-specific metrics
        """
        ...
