from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict


class DetectionResult(BaseModel):
    """
    Represents a single detection result from the detector.

    Attributes:
        box: List of floats representing bounding box [x1, y1, x2, y2].
        confidence: Detection confidence score (0.0 to 1.0).
        class_id: Numerical ID of the detected class.
        label: Human-readable label of the detected class.
        image: Optional numpy array of the cropped detection area.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    box: List[float]
    confidence: float
    class_id: int
    label: str
    image: Optional[np.ndarray] = None


class BaseDetector(ABC):
    """Abstract base class for all vehicle/plate detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in the given image.
        Returns a list of DetectionResult, sorted by confidence descending.
        """
        pass
