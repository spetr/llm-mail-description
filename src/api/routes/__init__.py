"""API routes."""

from fastapi import APIRouter

from src.api.routes.analyze import router as analyze_router
from src.api.routes.health import router as health_router

# Aggregate all routers
router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(analyze_router, tags=["analyze"])

__all__ = ["router"]
