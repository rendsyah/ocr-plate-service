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
    camera = getattr(request.app.state, "camera", None)
    cleanup_task = getattr(request.app.state, "cleanup_task", None)

    pipeline_ready = pipeline is not None
    storage_ready = storage is not None
    camera_ready = camera is not None
    cleanup_worker_active = cleanup_task is not None and not cleanup_task.done()

    # Determine overall status
    is_healthy = pipeline_ready and storage_ready and camera_ready
    status = "healthy" if is_healthy else "unhealthy"

    data = HealthData(
        status=status,
        app_name=settings.app_name,
        ocr_engine=settings.ocr_engine.value,
        ocr_device="cuda" if settings.ocr_use_gpu else "cpu",
        detector_type=settings.detector_type,
        detector_device="cuda" if settings.detector_use_gpu else "cpu",
        pipeline_ready=pipeline_ready,
        storage_ready=storage_ready,
        camera_ready=camera_ready,
        cleanup_worker_active=cleanup_worker_active,
    )

    return SuccessResponse(data=data)
