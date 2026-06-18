from src.config import CameraType, get_settings

from .base import BaseCamera
from .snapshot import SnapshotCamera


def get_camera() -> BaseCamera:
    """
    Factory function to create a camera implementation based on settings.
    """
    settings = get_settings()

    if settings.camera_type == CameraType.SNAPSHOT:
        return SnapshotCamera(
            url=settings.camera_source,
            timeout=settings.camera_timeout,
            retries=settings.camera_retries,
        )
    else:
        raise ValueError(f"Unsupported camera type: {settings.camera_type}")
