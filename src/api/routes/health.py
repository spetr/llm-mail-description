"""Health check endpoints."""

from typing import Any

from fastapi import APIRouter

from src.api.dependencies import AppStateDep

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """
    Basic liveness check.

    Returns 200 if the API is running.
    """
    return {"status": "ok"}


@router.get("/ready")
async def ready(app_state: AppStateDep) -> dict[str, Any]:
    """
    Readiness check.

    Returns 200 if the API is ready to handle requests.
    Checks batch processor and inference backend status.
    """
    ready_status = True
    details: dict[str, Any] = {}

    # Check batch processor
    if app_state.batch_processor:
        stats = app_state.batch_processor.get_stats()
        details["batch_processor"] = {
            "running": stats.get("running", False),
            "queue_size": stats.get("queue_size", 0),
        }
        if not stats.get("running"):
            ready_status = False
    else:
        details["batch_processor"] = {"running": False}
        ready_status = False

    # Check inference backend
    if app_state.inference_backend:
        backend_healthy = await app_state.inference_backend.health_check()
        details["inference_backend"] = {"healthy": backend_healthy}
        if not backend_healthy:
            ready_status = False
    else:
        details["inference_backend"] = {"healthy": False}
        ready_status = False

    return {
        "ready": ready_status,
        "details": details,
    }


@router.get("/stats")
async def stats(app_state: AppStateDep) -> dict[str, Any]:
    """
    Get service statistics.

    Returns detailed stats about batch processor and inference backend.
    """
    result: dict[str, Any] = {}

    if app_state.batch_processor:
        result["batch_processor"] = app_state.batch_processor.get_stats()

    if app_state.inference_backend:
        result["inference_backend"] = app_state.inference_backend.get_stats()

    return result
