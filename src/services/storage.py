import asyncio
import datetime
import os
import shutil

import cv2
import numpy as np

from src.config import get_settings
from src.domain import PlatePrediction
from src.utils import logger


class StorageService:
    """
    Handles all disk I/O operations for snapshots and evidence.
    Abstracts folder structures and naming conventions.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = logger.bind(context=self.__class__.__name__)

    def _generate_paths(
        self, plate_number: str, confidence: float, is_valid: bool
    ) -> tuple[str, str]:
        """Internal helper to generate relative and full paths."""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        sub_folder = "valid" if is_valid else "invalid"
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        clean_plate = "".join(e for e in plate_number if e.isalnum())
        filename = f"{timestamp}_{clean_plate}_{confidence:.2f}.jpg"

        rel_path = os.path.join(date_str, sub_folder, filename)
        full_path = os.path.join(self.settings.snapshot_dir, rel_path)

        return full_path, filename

    def _write_image(self, image: np.ndarray, full_path: str):
        """Internal worker to perform the actual I/O for any image type."""
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            cv2.imwrite(full_path, image)
            self.logger.bind(path=full_path).debug("Image saved to disk")
        except Exception as e:
            self.logger.bind(path=full_path).error(f"Failed to save image: {e}")

    def save_snapshot(self, image: np.ndarray, prediction: PlatePrediction) -> str:
        """
        Saves a snapshot.
        Returns the relative path immediately and performs the I/O in the background.
        """
        full_path, filename = self._generate_paths(
            prediction.plate_number, prediction.confidence, prediction.is_valid
        )

        # Offload blocking I/O to a thread and run it in the background
        if self.settings.save_snapshots:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(asyncio.to_thread(self._write_image, image, full_path))
            except RuntimeError:
                self._write_image(image, full_path)

        return filename

    def save_debug_image(
        self, image: np.ndarray, category: str, filename: str
    ) -> str | None:
        """
        Saves a debug image in a specific category folder.
        Runs in background if an event loop is available.
        """
        if self.settings.app_env != "development":
            return None

        debug_path = os.path.join(self.settings.debug_dir, category)
        file_path = os.path.join(debug_path, filename)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._write_image, image, file_path))
        except RuntimeError:
            self._write_image(image, file_path)

        return filename

    def cleanup_old_snapshots(self):
        """Deletes snapshot directories older than retention_days."""
        now = datetime.datetime.now()
        retention_delta = datetime.timedelta(days=self.settings.retention_days)
        target_dirs = [self.settings.snapshot_dir, self.settings.debug_dir]

        self.logger.bind(retention_days=self.settings.retention_days).info(
            "Starting cleanup"
        )

        for base_dir in target_dirs:
            if not os.path.exists(base_dir):
                continue

            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                try:
                    file_time = datetime.datetime.fromtimestamp(
                        os.path.getmtime(item_path)
                    )
                    if now - file_time > retention_delta:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            self.logger.bind(path=item_path).info(
                                "Deleted old directory"
                            )
                        else:
                            os.remove(item_path)
                            self.logger.bind(path=item_path).info("Deleted old file")
                except Exception as e:
                    self.logger.bind(path=item_path).warning(f"Failed to delete: {e}")

    async def run_cleanup_worker(self):
        """Background worker for periodic cleanup."""
        # Initial wait to allow the system to stabilize during startup
        await asyncio.sleep(10)
        while True:
            try:
                self.cleanup_old_snapshots()
            except Exception as e:
                self.logger.error(f"Periodic cleanup error: {e}")
            await asyncio.sleep(24 * 3600)
