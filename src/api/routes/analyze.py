"""Email analysis endpoint."""

import time
from typing import Any

import structlog
from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.api.dependencies import BatchProcessorDep
from src.api.exceptions import InferenceError, ServiceUnavailableError

logger = structlog.get_logger()

router = APIRouter()


class AnalyzeRequest(BaseModel):
    """Request body for email analysis."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Email content in plain text or markdown format",
    )


class AnalyzeResponse(BaseModel):
    """
    Response wrapper for email analysis.

    The 'analysis' field contains the dynamic EmailAnalysis model.
    """

    analysis: dict[str, Any] = Field(
        ...,
        description="Email analysis result matching the configured schema",
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds",
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze an email",
    description="Submit an email for analysis. Returns structured JSON with summary, keywords, categories, and tone.",
)
async def analyze(
    request: AnalyzeRequest,
    batch_processor: BatchProcessorDep,
) -> AnalyzeResponse:
    """
    Analyze a single email.

    The request is queued and batched with other concurrent requests
    for efficient inference. Response is returned when processing completes.
    """
    start_time = time.time()

    # Check if processor is running
    stats = batch_processor.get_stats()
    if not stats.get("running"):
        raise ServiceUnavailableError("Batch processor is not running")

    try:
        # Submit to batch processor and wait for result
        result = await batch_processor.submit(request.content)

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "email_analyzed",
            content_length=len(request.content),
            processing_time_ms=round(processing_time_ms, 2),
        )

        return AnalyzeResponse(
            analysis=result.model_dump(),
            processing_time_ms=round(processing_time_ms, 2),
        )

    except RuntimeError as e:
        logger.error("analysis_failed", error=str(e))
        raise ServiceUnavailableError(str(e)) from e

    except Exception as e:
        logger.error("analysis_failed", error=str(e))
        raise InferenceError(f"Analysis failed: {e}") from e
