"""Abstract protocol for batch processors.

This abstraction allows swapping implementations:
- MicroBatchProcessor: In-memory asyncio.Queue (current)
- RabbitMQProcessor: RabbitMQ-based (future)
- RedisProcessor: Redis Streams (future)
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BatchProcessor(ABC):
    """
    Abstract base for batch processors.

    Implementations handle queuing requests, batching them,
    and dispatching to inference backend.
    """

    @abstractmethod
    async def start(self) -> None:
        """Start the batch processor background tasks."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the batch processor and cleanup resources."""
        ...

    @abstractmethod
    async def submit(self, email_content: str) -> BaseModel:
        """
        Submit an email for analysis.

        Args:
            email_content: Raw email text (markdown or plain text)

        Returns:
            EmailAnalysis model instance with analysis results
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get current processor statistics.

        Returns:
            Dict with keys like: queue_size, processed_count, avg_batch_size
        """
        ...
