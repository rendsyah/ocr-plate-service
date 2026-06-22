from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict


class OCRResult(BaseModel):
    """
    Represents a single OCR recognition result.

    Attributes:
        text: The recognized alphanumeric string.
        confidence: Recognition confidence score (0.0 to 1.0).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    confidence: float


class BaseOCR(ABC):
    """Abstract base class for all OCR engines."""

    @abstractmethod
    def recognize(self, image: np.ndarray) -> Optional[OCRResult]:
        """
        Recognize text from a cropped plate image.
        Returns an OCRResult or None if no text is found.
        """
        pass
