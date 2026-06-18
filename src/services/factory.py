from src.config import DetectorType, NormalizerCountry, OCREngine, get_settings
from src.domain import BaseDetector, BaseNormalizer, BaseOCR, IndonesianPlateNormalizer

from .detector import YOLODetector
from .ocr import PaddleOCR
from .storage import StorageService


def get_detector() -> BaseDetector:
    """
    Factory function to create a detector instance based on settings.
    """
    settings = get_settings()

    if settings.detector_type == DetectorType.YOLO:
        return YOLODetector(
            model_path=settings.detector_model_path,
            confidence=settings.min_detection_confidence,
            device="cuda" if settings.detector_use_gpu else "cpu",
        )
    else:
        raise ValueError(f"Unsupported detector type: {settings.detector_type}")


def get_ocr() -> BaseOCR:
    """
    Factory function to create an OCR engine instance based on settings.
    """
    settings = get_settings()

    if settings.ocr_engine == OCREngine.PADDLEOCR:
        return PaddleOCR(use_gpu=settings.ocr_use_gpu)
    else:
        raise ValueError(f"Unsupported OCR engine: {settings.ocr_engine}")


def get_normalizer() -> BaseNormalizer:
    """
    Factory function to create a plate normalizer based on settings.
    """
    settings = get_settings()

    if settings.normalizer_country == NormalizerCountry.INDONESIA:
        return IndonesianPlateNormalizer()
    else:
        raise ValueError(
            f"Unsupported country normalizer: {settings.normalizer_country}"
        )


def get_storage() -> StorageService:
    """
    Factory function to create a storage service instance.
    """
    return StorageService()
