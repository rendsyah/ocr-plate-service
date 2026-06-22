import asyncio
import datetime
import os

import cv2
import numpy as np

from src.config import get_settings
from src.domain import PlatePrediction
from src.utils import logger


class StorageService:
    """
    Handles all disk I/O operations for snapshots and preprocessed images.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = logger.bind(context=self.__class__.__name__)

    def save_results(
        self,
        image: np.ndarray,
        prediction: PlatePrediction,
        preprocess_image: np.ndarray,
    ) -> tuple[str, str]:
        """
        Save snapshot and preprocessed image together.
        Both are controlled by the save_snapshots setting.
        Returns (snapshot_filename, preprocess_filename).
        """
        snapshot_filename = ""
        preprocess_filename = ""

        if not self.settings.save_snapshots:
            return snapshot_filename, preprocess_filename

        snapshot_path, snapshot_filename = self._snapshot_path(prediction)

        proc_suffix = f"{int(datetime.datetime.now().timestamp() * 1000)}.jpg"
        preprocess_filename = f"proc_{proc_suffix}"
        preprocess_path = self._preprocess_path(preprocess_filename)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._write, image, snapshot_path))
            loop.create_task(
                asyncio.to_thread(self._write, preprocess_image, preprocess_path)
            )
        except RuntimeError:
            self._write(image, snapshot_path)
            self._write(preprocess_image, preprocess_path)

        self.logger.bind(
            snapshot=snapshot_filename,
            preprocess=preprocess_filename,
        ).debug("Results saved")

        return snapshot_filename, preprocess_filename

    def _snapshot_path(self, prediction: PlatePrediction) -> tuple[str, str]:
        """Generate full path and filename for a snapshot image."""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        sub_folder = "valid" if prediction.is_valid else "invalid"
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        clean_plate = "".join(e for e in prediction.plate_number if e.isalnum())
        filename = f"{timestamp}_{clean_plate}_{prediction.confidence:.2f}.jpg"
        full_path = os.path.join(
            self.settings.snapshot_dir, date_str, sub_folder, filename
        )
        return full_path, filename

    def _preprocess_path(self, filename: str) -> str:
        """Generate full path for a preprocessed image."""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.settings.preprocess_dir, date_str, filename)

    def _write(self, image: np.ndarray, path: str):
        """Sync disk I/O."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, image)
        except Exception as e:
            self.logger.bind(path=path).error(f"Failed to save image: {e}")
