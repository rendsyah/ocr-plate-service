from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain import PlatePrediction


@pytest.fixture
def mock_detector(mock_detection, mock_vehicle_detection):
    detector = MagicMock()
    detector.detect.return_value = [mock_detection, mock_vehicle_detection]
    return detector


@pytest.fixture
def mock_ocr(mock_ocr_result):
    ocr = MagicMock()
    ocr.recognize.return_value = mock_ocr_result
    return ocr


@pytest.fixture
def mock_normalizer():
    normalizer = MagicMock()
    normalizer.normalize.return_value = "BM6432YZ"
    normalizer.is_valid.return_value = True
    return normalizer


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.save_results.return_value = ("snap.jpg", "proc.jpg")
    return storage


@pytest.fixture
def pipeline(mock_detector, mock_ocr, mock_normalizer, mock_storage):
    from src.services.pipeline import OCRPipeline

    return OCRPipeline(mock_detector, mock_ocr, mock_normalizer, mock_storage)


class TestProcessFrame:
    def test_successful_pipeline(self, pipeline, mock_image):
        result = pipeline.process_frame(mock_image)
        assert isinstance(result, PlatePrediction)
        assert result.plate_number == "BM6432YZ"
        assert result.is_valid is True
        assert result.vehicle_type == "car"
        assert 0 <= result.confidence <= 1

    def test_no_detection_returns_none(self, pipeline, mock_image, mock_detector):
        mock_detector.detect.return_value = []
        result = pipeline.process_frame(mock_image)
        assert result is None

    def test_no_ocr_result_returns_none(self, pipeline, mock_image, mock_ocr):
        mock_ocr.recognize.return_value = None
        result = pipeline.process_frame(mock_image)
        assert result is None

    def test_low_ocr_confidence_returns_none(self, pipeline, mock_image, mock_ocr):
        mock_ocr.recognize.return_value.confidence = 0.3
        result = pipeline.process_frame(mock_image)
        assert result is None

    def test_empty_image_returns_none(self, pipeline):
        result = pipeline.process_frame(np.array([]))
        assert result is None

    def test_null_image_returns_none(self, pipeline):
        result = pipeline.process_frame(None)
        assert result is None

    def test_preprocess_small_image(self, pipeline, mock_image):
        small = np.zeros((20, 100, 3), dtype=np.uint8)
        with patch("src.services.pipeline.cv2.resize") as mock_resize:
            mock_resize.return_value = np.zeros((40, 200, 3), dtype=np.uint8)
            pipeline._preprocess(small)
            mock_resize.assert_called_once()

    def test_preprocess_large_no_resize(self, pipeline):
        large = np.zeros((100, 200, 3), dtype=np.uint8)
        result = pipeline._preprocess(large)
        assert result.shape == large.shape

    def test_no_plate_detection_returns_none(
        self, pipeline, mock_image, mock_detector, mock_vehicle_detection
    ):
        mock_detector.detect.return_value = [mock_vehicle_detection]
        result = pipeline.process_frame(mock_image)
        assert result is None

    def test_vehicle_type_unknown_when_only_plate(
        self, pipeline, mock_image, mock_detector, mock_detection
    ):
        mock_detector.detect.return_value = [mock_detection]
        result = pipeline.process_frame(mock_image)
        assert result is not None
        assert result.vehicle_type == "unknown"

    def test_vehicle_type_associated_by_containment(
        self, pipeline, mock_image, mock_detector, mock_detection
    ):
        far_vehicle = MagicMock()
        far_vehicle.class_id = 1
        far_vehicle.label = "motorcycle"
        far_vehicle.box = [500.0, 500.0, 600.0, 550.0]

        containing_vehicle = MagicMock()
        containing_vehicle.class_id = 2
        containing_vehicle.label = "car"
        containing_vehicle.box = [0.0, 0.0, 200.0, 100.0]

        mock_detector.detect.return_value = [
            mock_detection,
            far_vehicle,
            containing_vehicle,
        ]
        result = pipeline.process_frame(mock_image)
        assert result is not None
        assert result.vehicle_type == "car"

    def test_vehicle_type_closest_by_distance(
        self, pipeline, mock_image, mock_detector, mock_detection
    ):
        far_vehicle = MagicMock()
        far_vehicle.class_id = 2
        far_vehicle.label = "motorcycle"
        far_vehicle.box = [500.0, 500.0, 600.0, 550.0]

        near_vehicle = MagicMock()
        near_vehicle.class_id = 1
        near_vehicle.label = "car"
        near_vehicle.box = [30.0, 30.0, 180.0, 90.0]

        mock_detector.detect.return_value = [mock_detection, far_vehicle, near_vehicle]
        result = pipeline.process_frame(mock_image)
        assert result is not None
        assert result.vehicle_type == "car"

    def test_metadata_includes_timing(self, pipeline, mock_image):
        result = pipeline.process_frame(mock_image)
        metrics = result.metadata.get("metrics", {})
        assert "total_ms" in metrics
        assert metrics["total_ms"] > 0
