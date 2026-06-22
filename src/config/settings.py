from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class OCREngine(str, Enum):
    PADDLEOCR = "paddleocr"


class DetectorType(str, Enum):
    YOLO = "yolo"


class NormalizerCountry(str, Enum):
    INDONESIA = "indonesia"


class Settings(BaseSettings):
    # App Settings
    app_name: str = "ocr-plate-service"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    log_level: str = "INFO"

    # OCR Settings
    ocr_engine: OCREngine = OCREngine.PADDLEOCR
    ocr_detection_model: str = "PP-OCRv6_small_det"
    ocr_recognition_model: str = "PP-OCRv6_small_rec"
    normalizer_country: NormalizerCountry = NormalizerCountry.INDONESIA
    min_ocr_confidence: float = 0.5

    # Inference Settings
    detector_type: DetectorType = DetectorType.YOLO
    plate_model_path: str = "models/license_plate.pt"
    vehicle_model_path: str = "models/vehicle.pt"
    min_detection_confidence: float = 0.5

    # Storage Settings
    save_snapshots: bool = True
    snapshot_dir: str = "storage/snapshots"
    preprocess_dir: str = "storage/preprocess"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
