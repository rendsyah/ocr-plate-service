from fastapi import APIRouter, Request

from src.api.v1.schemas import HealthData, SuccessResponse
from src.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=SuccessResponse[HealthData])
async def health_check(request: Request):
    """Health check endpoint to verify service and model status."""
    pipeline = getattr(request.app.state, "pipeline", None)
    storage = getattr(request.app.state, "storage", None)

    pipeline_ready = pipeline is not None
    storage_ready = storage is not None

    # Determine overall status
    is_healthy = pipeline_ready and storage_ready
    status = "healthy" if is_healthy else "unhealthy"

    data = HealthData(
        status=status,
        app_name=settings.app_name,
        ocr_engine=settings.ocr_engine.value,
        ocr_device="cpu",
        detector_type=settings.detector_type,
        detector_device="cpu",
        pipeline_ready=pipeline_ready,
        storage_ready=storage_ready,
    )

    return SuccessResponse(data=data)
