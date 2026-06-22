import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def valid_jpeg() -> bytes:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def client():
    """Create test client with mocked pipeline in app state."""
    with patch("src.main.setup_app_logging"):
        with patch("src.main.get_detector"):
            with patch("src.main.get_ocr"):
                with patch("src.main.get_normalizer"):
                    with patch("src.main.get_storage"):
                        from src.main import app
                        from fastapi.testclient import TestClient

                        with TestClient(app) as c:
                            pipeline = MagicMock()
                            app.state.pipeline = pipeline
                            yield c, pipeline


class TestHealth:
    def test_health_returns_200(self, client):
        c, _ = client
        response = c.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "healthy"
        assert body["data"]["pipeline_ready"] is True
        assert body["data"]["storage_ready"] is True

    def test_health_contains_pipeline_status(self, client):
        c, _ = client
        response = c.get("/api/v1/health")
        assert response.status_code == 200


class TestPredict:
    def test_predict_no_file_returns_422(self, client):
        c, _ = client
        response = c.post("/api/v1/ocr/predict")
        assert response.status_code == 422

    def test_predict_with_image_returns_prediction(self, client, valid_jpeg):
        c, pipeline = client
        metadata = {
            "metrics": {
                "detection_ms": 100.0,
                "preprocess_ms": 5.0,
                "ocr_ms": 200.0,
                "normalize_ms": 1.0,
                "total_ms": 306.0,
            },
            "snapshot_filename": "snap.jpg",
            "preprocess_filename": "proc.jpg",
        }
        pipeline.process_frame.return_value = MagicMock(
            plate_number="BM6432YZ",
            vehicle_type="motorcycle",
            confidence=0.94,
            detection_conf=0.95,
            ocr_conf=0.92,
            is_valid=True,
            box=[10.0, 20.0, 150.0, 80.0],
            metadata=metadata,
            model_dump=lambda metadata=metadata: {
                "plate_number": "BM6432YZ",
                "vehicle_type": "motorcycle",
                "confidence": 0.94,
                "detection_conf": 0.95,
                "ocr_conf": 0.92,
                "is_valid": True,
                "box": [10.0, 20.0, 150.0, 80.0],
                "metadata": metadata,
            },
        )

        response = c.post(
            "/api/v1/ocr/predict",
            files={"file": ("test.jpg", valid_jpeg, "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        data = body["data"]
        assert data["plate_number"] == "BM6432YZ"
        assert data["vehicle_type"] == "motorcycle"
        assert data["is_valid"] is True

    def test_predict_test_endpoint(self, client):
        c, pipeline = client
        metadata = {
            "metrics": {
                "detection_ms": 100.0,
                "preprocess_ms": 5.0,
                "ocr_ms": 200.0,
                "normalize_ms": 1.0,
                "total_ms": 306.0,
            },
            "snapshot_filename": "snap.jpg",
            "preprocess_filename": "proc.jpg",
        }
        pipeline.process_frame.return_value = MagicMock(
            plate_number="BM6432YZ",
            vehicle_type="motorcycle",
            confidence=0.94,
            detection_conf=0.95,
            ocr_conf=0.92,
            is_valid=True,
            box=[10.0, 20.0, 150.0, 80.0],
            metadata=metadata,
            model_dump=lambda metadata=metadata: {
                "plate_number": "BM6432YZ",
                "vehicle_type": "motorcycle",
                "confidence": 0.94,
                "detection_conf": 0.95,
                "ocr_conf": 0.92,
                "is_valid": True,
                "box": [10.0, 20.0, 150.0, 80.0],
                "metadata": metadata,
            },
        )

        response = c.post("/api/v1/ocr/predict-test")
        assert response.status_code == 200
