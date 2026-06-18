from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class OCREngine(str, Enum):
    PADDLEOCR = "paddleocr"


class DetectorType(str, Enum):
    YOLO = "yolo"


class NormalizerCountry(str, Enum):
    INDONESIA = "indonesia"


class CameraType(str, Enum):
    SNAPSHOT = "snapshot"


class Settings(BaseSettings):
    # App Settings
    app_name: str = "ocr-plate-service"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    log_level: str = "INFO"

    # OCR Settings
    ocr_engine: OCREngine = OCREngine.PADDLEOCR
    ocr_use_gpu: bool = False
    ocr_detection_model: str = "PP-OCRv6_small_det"
    ocr_recognition_model: str = "PP-OCRv6_small_rec"
    normalizer_country: NormalizerCountry = NormalizerCountry.INDONESIA
    min_ocr_confidence: float = 0.5

    # Inference Settings
    detector_type: DetectorType = DetectorType.YOLO
    detector_model_path: str = "models/license_plate.pt"
    detector_use_gpu: bool = False
    min_detection_confidence: float = 0.5

    # Camera & Ingestion
    camera_type: CameraType = CameraType.SNAPSHOT
    camera_source: str = "http://localhost:8080/snapshot"  # Camera Snapshot URL
    camera_timeout: float = 5.0
    camera_retries: int = 3

    # Storage Settings
    save_snapshots: bool = True
    snapshot_dir: str = "storage/snapshots"
    debug_dir: str = "storage/debug"
    retention_days: int = 7

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
