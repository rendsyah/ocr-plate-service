# fmt: off

from pydantic import BaseModel, Field


class HealthData(BaseModel):
    status: str = Field(..., description="Service health status (healthy or unhealthy)")
    app_name: str = Field(..., description="Application name")
    ocr_engine: str = Field(..., description="Active OCR engine")
    ocr_device: str = Field(..., description="OCR computing device (cpu or cuda)")
    detector_type: str = Field(..., description="Active detector type")
    detector_device: str = Field(..., description="Detector computing device (cpu or cuda)")
    pipeline_ready: bool = Field(..., description="Whether the AI pipeline is initialized")
    storage_ready: bool = Field(..., description="Whether the storage service is initialized")
    camera_ready: bool = Field(..., description="Whether the camera implementation is initialized")
    cleanup_worker_active: bool = Field(..., description="Whether the background cleanup worker is running")
