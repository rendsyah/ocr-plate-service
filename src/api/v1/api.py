from fastapi import APIRouter

from src.api.v1.endpoints import health, ocr

api_router = APIRouter()

api_router.include_router(health.router, tags=["Status"])
api_router.include_router(ocr.router, prefix="/ocr", tags=["OCR"])
