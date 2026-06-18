from fastapi import Depends, Request

from src.api.v1.services import OCRApiService
from src.camera import BaseCamera
from src.services import OCRPipeline, StorageService


def get_pipeline(request: Request) -> OCRPipeline:
    """Dependency to get the initialized OCR Pipeline from app state."""
    return request.app.state.pipeline


def get_camera(request: Request) -> BaseCamera:
    """Dependency to get the initialized Camera implementation from app state."""
    return request.app.state.camera


def get_storage(request: Request) -> StorageService:
    """Dependency to get the StorageService from app state."""
    return request.app.state.storage


def get_ocr_api_service(
    pipeline: OCRPipeline = Depends(get_pipeline),
    camera: BaseCamera = Depends(get_camera),
    storage: StorageService = Depends(get_storage),
) -> OCRApiService:
    """Dependency to get the API-specific OCR service."""
    return OCRApiService(pipeline, camera, storage)
