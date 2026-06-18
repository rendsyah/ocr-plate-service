from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from src.domain import BaseDetector, DetectionResult
from src.utils import logger


class YOLODetector(BaseDetector):
    """
    YOLO-based vehicle and license plate detector.
    Optimized for production with GPU warmup and efficient tensor handling.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to the .pt model file.
            confidence: Minimum confidence threshold.
            device: Inference device (e.g., 'cpu', 'cuda', '0').
        """
        self.confidence = confidence
        self.device = device
        self.logger = logger.bind(context=self.__class__.__name__)
        try:
            self.logger.info(f"Loading YOLO model from {model_path}...")
            self.model = YOLO(model_path)

            if device:
                self.model.to(device)

            # GPU Warmup: Crucial for production to avoid the slow "first inference" delay.
            is_cuda = "cuda" in str(self.device or "").lower() or (
                device is None and self.model.device.type == "cuda"
            )

            if is_cuda:
                self.logger.info("Performing YOLO GPU warmup...")
                warmup_img = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model.predict(warmup_img, verbose=False)
                self.logger.success("GPU warmup completed.")

            self.logger.success(
                f"YOLO model loaded successfully on {self.model.device.type}."
            )
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run YOLO inference on the given image.
        Returns a list of DetectionResult sorted by confidence descending.
        """
        if image is None or image.size == 0:
            return []

        try:
            # Run inference
            results = self.model.predict(
                source=image,
                conf=self.confidence,
                verbose=False,
                device=self.device,
            )

            detections: List[DetectionResult] = []
            img_h, img_w = image.shape[:2]

            for result in results:
                if result.boxes is None:
                    continue

                # Batch move boxes to CPU for efficiency if they aren't already
                boxes = result.boxes

                for box in boxes:
                    # Extract coordinates safely
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords

                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls]

                    # Clamp coordinates to image boundaries to prevent indexing errors
                    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                    ix2, iy2 = min(img_w, int(x2)), min(img_h, int(y2))

                    # Crop detection region
                    crop = image[iy1:iy2, ix1:ix2]

                    if crop.size == 0:
                        continue

                    detections.append(
                        DetectionResult(
                            box=[float(x1), float(y1), float(x2), float(y2)],
                            confidence=conf,
                            class_id=cls,
                            label=label,
                            image=crop,
                        )
                    )

            # Sort by confidence descending
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections

        except Exception as e:
            self.logger.error(f"YOLO inference error: {e}")
            return []
