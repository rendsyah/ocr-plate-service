"""Service layer for OCR Plate Service."""

from .detector import YOLODetector
from .factory import get_detector, get_normalizer, get_ocr, get_storage
from .ocr import PaddleOCR
from .pipeline import OCRPipeline
from .storage import StorageService

__all__ = [
    "YOLODetector",
    "PaddleOCR",
    "OCRPipeline",
    "StorageService",
    "get_detector",
    "get_ocr",
    "get_normalizer",
    "get_storage",
]
