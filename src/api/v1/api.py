from fastapi import APIRouter

from src.api.v1.endpoints import health_router, ocr_router

api_router = APIRouter()

api_router.include_router(health_router, tags=["Status"])
api_router.include_router(ocr_router, prefix="/ocr", tags=["OCR"])
