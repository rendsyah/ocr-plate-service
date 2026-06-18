"""Configuration module for OCR Plate Service."""

from .openapi import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from .settings import (
    CameraType,
    DetectorType,
    NormalizerCountry,
    OCREngine,
    get_settings,
)

__all__ = [
    "get_settings",
    "OCREngine",
    "DetectorType",
    "NormalizerCountry",
    "CameraType",
    "APP_TITLE",
    "APP_VERSION",
    "APP_DESCRIPTION",
]
