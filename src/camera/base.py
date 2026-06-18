from abc import ABC, abstractmethod

import numpy as np


class BaseCamera(ABC):
    """Base class for camera implementations."""

    @abstractmethod
    async def capture(self) -> np.ndarray:
        """
        Capture a single frame from the camera.

        Returns:
            np.ndarray: The captured frame as a numpy array (BGR).

        Raises:
            Exception: If capture fails.
        """
        pass
