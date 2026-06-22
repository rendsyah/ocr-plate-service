"""Configuration module for OCR Plate Service."""

from .openapi import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from .settings import (
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
    "APP_TITLE",
    "APP_VERSION",
    "APP_DESCRIPTION",
]
