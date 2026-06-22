# fmt: off

from typing import Optional

from pydantic import BaseModel, Field


class OCRMetrics(BaseModel):
    detection_ms: float = Field(..., description="Time taken for detection in milliseconds")
    preprocess_ms: float = Field(..., description="Time taken for preprocessing in milliseconds")
    ocr_ms: float = Field(..., description="Time taken for OCR recognition in milliseconds")
    normalize_ms: float = Field(..., description="Time taken for normalization in milliseconds")
    total_ms: float = Field(..., description="Total processing time in milliseconds")


class OCRMetadata(BaseModel):
    metrics: OCRMetrics


class OCRData(BaseModel):
    plate_number: str = Field(..., description="The normalized license plate number")
    vehicle_type: str = Field(..., description="Type of vehicle detected (car, motorcycle, etc.)")
    confidence: float = Field(..., description="Average confidence score (0.0 to 1.0)")
    ocr_conf: float = Field(..., description="OCR specific confidence score")
    detection_conf: float = Field(..., description="Detection specific confidence score")
    is_valid: bool = Field(..., description="Whether the plate format is valid for the country")
    snapshot_filename: Optional[str] = Field(None, description="Filename of the saved snapshot")
    preprocess_filename: Optional[str] = Field(None, description="Filename of the saved preprocessed plate image")
    sample_used: Optional[str] = Field(None, description="Filename of the sample used (for test endpoints)")
    metadata: OCRMetadata
