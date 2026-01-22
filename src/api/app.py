"""FastAPI application factory."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI

from src.api.dependencies import _app_state
from src.api.routes import router
from src.batch.processor import MicroBatchProcessor
from src.config.settings import get_settings
from src.inference.prompt import PromptManager
from src.inference.triton import TritonBackend
from src.schema.loader import load_schema

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Initializes resources on startup and cleans up on shutdown.
    """
    settings = get_settings()
    logger.info("application_starting", environment=settings.environment)

    # Load schema
    logger.info("loading_schema", path=str(settings.schema_path))
    email_analysis_model, schema_config = load_schema(settings.schema_path)

    # Initialize prompt manager
    prompt_manager = PromptManager(settings.prompts_path)
    prompt_manager.load_templates()

    # Initialize inference backend
    backend = TritonBackend(
        triton_urls=settings.app.inference.triton_urls,
        model_name=settings.app.inference.model_name,
        email_analysis_model=email_analysis_model,
        schema_config=schema_config,
        prompt_manager=prompt_manager,
        timeout_seconds=settings.app.inference.timeout_seconds,
        max_input_tokens=settings.app.model.max_input_tokens,
    )
    await backend.initialize()

    # Initialize batch processor
    batch_processor = MicroBatchProcessor(
        inference_backend=backend,
        max_batch_size=settings.app.batching.max_batch_size,
        max_wait_ms=settings.app.batching.max_wait_ms,
    )
    await batch_processor.start()

    # Store in app state
    _app_state.batch_processor = batch_processor
    _app_state.inference_backend = backend

    logger.info("application_started")

    yield

    # Shutdown
    logger.info("application_stopping")

    await batch_processor.stop()
    await backend.shutdown()

    logger.info("application_stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="MailBrain",
        description="Email analysis service using local LLM with constrained JSON output",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
    )

    # Include routes
    app.include_router(router)

    return app


# For uvicorn
app = create_app()
