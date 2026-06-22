from fastapi import Depends, Request

from src.api.v1.services import OCRApiService
from src.services import OCRPipeline


def get_pipeline(request: Request) -> OCRPipeline:
    """Dependency to get the initialized OCR Pipeline from app state."""
    return request.app.state.pipeline


def get_ocr_api_service(
    pipeline: OCRPipeline = Depends(get_pipeline),
) -> OCRApiService:
    """Dependency to get the API-specific OCR service."""
    return OCRApiService(pipeline)
