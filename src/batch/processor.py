"""Micro-batch processor using asyncio.Queue."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from pydantic import BaseModel

from src.batch.protocol import BatchProcessor
from src.inference.protocol import InferenceBackend

logger = structlog.get_logger()


@dataclass
class PendingRequest:
    """A request waiting in the queue."""

    email_content: str
    future: asyncio.Future[BaseModel]
    submitted_at: float = field(default_factory=time.time)


@dataclass
class ProcessorStats:
    """Statistics for the batch processor."""

    total_requests: int = 0
    total_batches: int = 0
    total_errors: int = 0

    @property
    def avg_batch_size(self) -> float:
        """Average number of requests per batch."""
        if self.total_batches == 0:
            return 0.0
        return self.total_requests / self.total_batches


class MicroBatchProcessor(BatchProcessor):
    """
    In-memory batch processor using asyncio.Queue.

    Collects incoming requests and batches them based on:
    - max_batch_size: Maximum requests per batch
    - max_wait_ms: Maximum time to wait for more requests

    Thread-safe and fully async.
    """

    def __init__(
        self,
        inference_backend: InferenceBackend,
        max_batch_size: int = 16,
        max_wait_ms: int = 50,
    ) -> None:
        self._backend = inference_backend
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_ms / 1000.0

        self._queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._process_task: asyncio.Task[None] | None = None
        self._running = False
        self._stats = ProcessorStats()

    async def start(self) -> None:
        """Start the background processing loop."""
        if self._running:
            return

        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(
            "batch_processor_started",
            max_batch_size=self._max_batch_size,
            max_wait_ms=int(self._max_wait_seconds * 1000),
        )

    async def stop(self) -> None:
        """Stop the processor and wait for pending requests."""
        if not self._running:
            return

        self._running = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.info("batch_processor_stopped", stats=self.get_stats())

    async def submit(self, email_content: str) -> BaseModel:
        """
        Submit an email for analysis.

        Blocks until the batch containing this request is processed.
        """
        if not self._running:
            raise RuntimeError("Batch processor is not running")

        future: asyncio.Future[BaseModel] = asyncio.Future()
        request = PendingRequest(email_content=email_content, future=future)

        await self._queue.put(request)
        logger.debug("request_queued", queue_size=self._queue.qsize())

        return await future

    def get_stats(self) -> dict[str, Any]:
        """Get current processor statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "total_requests": self._stats.total_requests,
            "total_batches": self._stats.total_batches,
            "total_errors": self._stats.total_errors,
            "avg_batch_size": round(self._stats.avg_batch_size, 2),
            "running": self._running,
        }

    async def _process_loop(self) -> None:
        """Main processing loop - collects and processes batches."""
        while self._running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("batch_processing_error", error=str(e))
                await asyncio.sleep(0.1)  # Avoid tight loop on errors

    async def _collect_batch(self) -> list[PendingRequest]:
        """
        Collect requests into a batch.

        Waits for max_wait_ms or until max_batch_size is reached.
        """
        batch: list[PendingRequest] = []
        deadline = time.time() + self._max_wait_seconds

        # Wait for at least one request
        try:
            first_request = await asyncio.wait_for(
                self._queue.get(),
                timeout=1.0,  # Check running state periodically
            )
            batch.append(first_request)
        except asyncio.TimeoutError:
            return []

        # Collect more requests until deadline or max size
        while len(batch) < self._max_batch_size:
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                break

            try:
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=remaining_time,
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        return batch

    async def _process_batch(self, batch: list[PendingRequest]) -> None:
        """Process a batch of requests through the inference backend."""
        start_time = time.time()
        emails = [req.email_content for req in batch]

        logger.info("batch_processing_start", batch_size=len(batch))

        try:
            results = await self._backend.analyze_batch(emails)

            # Distribute results to waiting futures
            for request, result in zip(batch, results):
                if not request.future.done():
                    request.future.set_result(result)

            self._stats.total_requests += len(batch)
            self._stats.total_batches += 1

            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "batch_processing_complete",
                batch_size=len(batch),
                duration_ms=round(duration_ms, 2),
            )

        except Exception as e:
            # Set exception on all pending futures
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

            self._stats.total_errors += 1
            logger.error("batch_processing_failed", batch_size=len(batch), error=str(e))
