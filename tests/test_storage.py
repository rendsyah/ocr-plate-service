from unittest.mock import patch

import numpy as np
import pytest

from src.domain import PlatePrediction


@pytest.fixture
def prediction() -> PlatePrediction:
    return PlatePrediction(
        plate_number="BM6432YZ",
        vehicle_type="motorcycle",
        confidence=0.94,
        detection_conf=0.95,
        ocr_conf=0.92,
        is_valid=True,
        box=[10.0, 20.0, 150.0, 80.0],
    )


@pytest.fixture
def storage():
    with patch("src.services.storage.cv2.imwrite"):
        with patch("src.services.storage.os.makedirs"):
            from src.services.storage import StorageService

            svc = StorageService()
            svc.settings.save_snapshots = True
            yield svc


class TestSnapshotPath:
    def test_format(self, storage, prediction):
        with patch("src.services.storage.datetime.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.side_effect = [
                "2026-06-20",
                "123456",
            ]
            path, filename = storage._snapshot_path(prediction)
            assert filename == "123456_BM6432YZ_0.94.jpg"
            assert "storage/snapshots" in path
            assert "2026-06-20" in path
            assert "valid" in path

    def test_invalid_prediction(self, storage):
        invalid = PlatePrediction(
            plate_number="INVALID",
            vehicle_type="car",
            confidence=0.5,
            detection_conf=0.5,
            ocr_conf=0.5,
            is_valid=False,
            box=[0.0, 0.0, 10.0, 10.0],
        )
        with patch("src.services.storage.datetime.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.side_effect = [
                "2026-06-20",
                "123456",
            ]
            path, filename = storage._snapshot_path(invalid)
            assert "invalid" in path


class TestPreprocessPath:
    def test_format(self, storage):
        with patch("src.services.storage.datetime.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-06-20"
            path = storage._preprocess_path("proc_123.jpg")
            assert "storage/preprocess" in path
            assert "2026-06-20" in path
            assert path.endswith("proc_123.jpg")


class TestWrite:
    def test_creates_directories(self, storage):
        with patch("src.services.storage.cv2.imwrite") as mock_write:
            with patch("src.services.storage.os.makedirs") as mock_makedirs:
                image = np.zeros((10, 10, 3), dtype=np.uint8)
                storage._write(image, "storage/test/img.jpg")
                mock_makedirs.assert_called_once_with("storage/test", exist_ok=True)
                mock_write.assert_called_once()

    def test_handles_exception_gracefully(self, storage):
        with patch("src.services.storage.cv2.imwrite") as mock_write:
            mock_write.side_effect = PermissionError("denied")
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            storage._write(image, "storage/test/img.jpg")


class TestSaveResults:
    def test_disabled_when_save_snapshots_false(self, storage):
        storage.settings.save_snapshots = False
        result = storage.save_results(None, None, None)
        assert result == ("", "")

    def test_enabled_returns_filenames(self, storage, prediction):
        with patch("src.services.storage.cv2.imwrite"):
            with patch("src.services.storage.os.makedirs"):
                image = np.zeros((10, 10, 3), dtype=np.uint8)
                snap_file, proc_file = storage.save_results(image, prediction, image)
                assert snap_file != ""
                assert proc_file != ""
