import cv2
import httpx
import numpy as np
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils import logger

from .base import BaseCamera


class SnapshotCamera(BaseCamera):
    """Camera implementation for IP camera snapshots via HTTP."""

    def __init__(self, url: str, timeout: float = 5.0, retries: int = 3):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.logger = logger.bind(context=self.__class__.__name__)

    async def capture(self) -> np.ndarray:
        """
        Fetch a JPEG snapshot from the IP camera URL and decode it.
        Uses Tenacity for robust retries with exponential backoff.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
            retry=retry_if_exception_type((httpx.HTTPError, Exception)),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    try:
                        response = await client.get(self.url)
                        response.raise_for_status()

                        # Convert bytes to numpy array and decode image
                        image_bytes = np.frombuffer(response.content, dtype=np.uint8)
                        frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                        if frame is None:
                            raise ValueError("Failed to decode image snapshot")

                        return frame
                    except Exception as e:
                        self.logger.warning(
                            f"Snapshot attempt {attempt.retry_state.attempt_number} failed: {e}"
                        )
                        raise

        raise Exception(f"Failed to capture snapshot from {self.url}")
