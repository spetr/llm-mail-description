"""FastAPI dependency injection."""

from typing import Annotated

from fastapi import Depends

from src.batch.processor import MicroBatchProcessor
from src.inference.protocol import InferenceBackend


class AppState:
    """
    Application state container.

    Holds references to shared resources initialized at startup.
    """

    batch_processor: MicroBatchProcessor | None = None
    inference_backend: InferenceBackend | None = None


# Global state instance
_app_state = AppState()


def get_app_state() -> AppState:
    """Get the application state."""
    return _app_state


def get_batch_processor() -> MicroBatchProcessor:
    """Get the batch processor instance."""
    if _app_state.batch_processor is None:
        raise RuntimeError("Batch processor not initialized")
    return _app_state.batch_processor


AppStateDep = Annotated[AppState, Depends(get_app_state)]
BatchProcessorDep = Annotated[MicroBatchProcessor, Depends(get_batch_processor)]
