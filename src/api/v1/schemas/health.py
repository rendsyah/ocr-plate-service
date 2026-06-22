# fmt: off

from pydantic import BaseModel, Field


class HealthData(BaseModel):
    status: str = Field(..., description="Service health status (healthy or unhealthy)")
    app_name: str = Field(..., description="Application name")
    ocr_engine: str = Field(..., description="Active OCR engine")
    ocr_device: str = Field(..., description="OCR computing device")
    detector_type: str = Field(..., description="Active detector type")
    detector_device: str = Field(..., description="Detector computing device")
    pipeline_ready: bool = Field(..., description="Whether the AI pipeline is initialized")
    storage_ready: bool = Field(..., description="Whether the storage service is initialized")
