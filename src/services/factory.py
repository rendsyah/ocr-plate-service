from src.config import DetectorType, NormalizerCountry, OCREngine, get_settings
from src.domain import BaseDetector, BaseNormalizer, BaseOCR, IndonesianPlateNormalizer

from .detector import ParallelDetector, YOLODetector, _VEHICLE_CLASS_MAP
from .ocr import PaddleOCR
from .storage import StorageService


def get_detector() -> BaseDetector:
    """
    Factory function to create a parallel plate + vehicle detector.
    """
    settings = get_settings()

    if settings.detector_type != DetectorType.YOLO:
        raise ValueError(f"Unsupported detector type: {settings.detector_type}")

    plate_detector = YOLODetector(
        model_path=settings.plate_model_path,
        confidence_threshold=settings.min_detection_confidence,
        class_offset=0,
    )
    vehicle_detector = YOLODetector(
        model_path=settings.vehicle_model_path,
        confidence_threshold=settings.min_detection_confidence,
        class_map=_VEHICLE_CLASS_MAP,
    )

    return ParallelDetector([plate_detector, vehicle_detector])


def get_ocr() -> BaseOCR:
    """
    Factory function to create an OCR engine instance based on settings.
    """
    settings = get_settings()

    if settings.ocr_engine == OCREngine.PADDLEOCR:
        return PaddleOCR()
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
