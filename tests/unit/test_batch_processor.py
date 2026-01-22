"""Tests for batch processor."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from src.batch.processor import MicroBatchProcessor
from src.inference.protocol import InferenceBackend


class MockAnalysis(BaseModel):
    """Mock analysis result."""

    summary: str


class MockBackend(InferenceBackend):
    """Mock inference backend for testing."""

    def __init__(self) -> None:
        self.call_count = 0
        self.batch_sizes: list[int] = []

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def analyze_batch(self, emails: list[str]) -> list[BaseModel]:
        self.call_count += 1
        self.batch_sizes.append(len(emails))
        await asyncio.sleep(0.01)  # Simulate processing
        return [MockAnalysis(summary=f"Summary of: {e[:20]}") for e in emails]

    async def health_check(self) -> bool:
        return True

    def get_stats(self) -> dict[str, Any]:
        return {"call_count": self.call_count}


class TestMicroBatchProcessor:
    """Tests for MicroBatchProcessor."""

    @pytest.fixture
    def backend(self) -> MockBackend:
        """Create mock backend."""
        return MockBackend()

    @pytest.fixture
    async def processor(self, backend: MockBackend) -> MicroBatchProcessor:
        """Create processor with mock backend."""
        proc = MicroBatchProcessor(
            inference_backend=backend,
            max_batch_size=4,
            max_wait_ms=50,
        )
        await proc.start()
        yield proc
        await proc.stop()

    @pytest.mark.asyncio
    async def test_single_request(self, processor: MicroBatchProcessor) -> None:
        """Test processing a single request."""
        result = await processor.submit("Test email content")
        assert isinstance(result, MockAnalysis)
        assert "Test email" in result.summary

    @pytest.mark.asyncio
    async def test_batches_concurrent_requests(
        self, processor: MicroBatchProcessor, backend: MockBackend
    ) -> None:
        """Test that concurrent requests are batched."""
        # Submit multiple requests concurrently
        tasks = [
            asyncio.create_task(processor.submit(f"Email {i}"))
            for i in range(4)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        # Should have been processed in 1-2 batches
        assert backend.call_count <= 2

    @pytest.mark.asyncio
    async def test_max_batch_size_respected(
        self, backend: MockBackend
    ) -> None:
        """Test that max_batch_size is respected."""
        processor = MicroBatchProcessor(
            inference_backend=backend,
            max_batch_size=2,
            max_wait_ms=100,
        )
        await processor.start()

        try:
            # Submit 4 requests - should create at least 2 batches
            tasks = [
                asyncio.create_task(processor.submit(f"Email {i}"))
                for i in range(4)
            ]
            await asyncio.gather(*tasks)

            # All batches should be <= max_batch_size
            assert all(size <= 2 for size in backend.batch_sizes)
        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_stats(self, processor: MicroBatchProcessor) -> None:
        """Test stats reporting."""
        await processor.submit("Test email")

        stats = processor.get_stats()
        assert stats["running"] is True
        assert stats["total_requests"] >= 1
        assert stats["total_batches"] >= 1

    @pytest.mark.asyncio
    async def test_not_running_raises(self, backend: MockBackend) -> None:
        """Test that submitting to stopped processor raises."""
        processor = MicroBatchProcessor(
            inference_backend=backend,
            max_batch_size=4,
            max_wait_ms=50,
        )
        # Not started

        with pytest.raises(RuntimeError, match="not running"):
            await processor.submit("Test")

    @pytest.mark.asyncio
    async def test_error_propagation(self, backend: MockBackend) -> None:
        """Test that backend errors propagate to caller."""
        backend.analyze_batch = AsyncMock(  # type: ignore
            side_effect=RuntimeError("Backend error")
        )

        processor = MicroBatchProcessor(
            inference_backend=backend,
            max_batch_size=4,
            max_wait_ms=50,
        )
        await processor.start()

        try:
            with pytest.raises(RuntimeError, match="Backend error"):
                await processor.submit("Test")
        finally:
            await processor.stop()
