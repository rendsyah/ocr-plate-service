import numpy as np
import pytest

from src.domain.detector import DetectionResult
from src.domain.ocr import OCRResult
from src.domain.pipeline import IndonesianPlateNormalizer


@pytest.fixture
def normalizer() -> IndonesianPlateNormalizer:
    return IndonesianPlateNormalizer()


@pytest.fixture
def mock_image() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


@pytest.fixture
def mock_detection() -> DetectionResult:
    return DetectionResult(
        box=[10.0, 20.0, 150.0, 80.0],
        confidence=0.95,
        class_id=0,
        label="license_plate",
        image=np.ones((60, 140, 3), dtype=np.uint8) * 255,
    )


@pytest.fixture
def mock_ocr_result() -> OCRResult:
    return OCRResult(text="BM6432YZ", confidence=0.92)


@pytest.fixture
def mock_vehicle_detection() -> DetectionResult:
    return DetectionResult(
        box=[0.0, 0.0, 200.0, 100.0],
        confidence=0.90,
        class_id=1,
        label="car",
        image=np.ones((100, 200, 3), dtype=np.uint8) * 200,
    )
