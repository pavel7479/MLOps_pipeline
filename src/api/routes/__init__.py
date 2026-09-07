"""FastAPI route modules."""

from .health import router as health_router
from .model import router as model_router
from .predictions import router as predictions_router

__all__ = ["health_router", "model_router", "predictions_router"]
