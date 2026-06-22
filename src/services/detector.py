from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from src.config import get_settings
from src.domain import BaseDetector, DetectionResult
from src.utils import logger

CLASS_LICENSE_PLATE = 0
CLASS_CAR = 1
CLASS_MOTORCYCLE = 2
CLASS_TRUCK = 3
CLASS_BUS = 4

VEHICLE_CLASS_IDS = {
    CLASS_CAR,
    CLASS_MOTORCYCLE,
    CLASS_TRUCK,
    CLASS_BUS,
}

_VEHICLE_CLASS_MAP: Dict[int, int] = {
    2: CLASS_CAR,
    3: CLASS_MOTORCYCLE,
    7: CLASS_TRUCK,
    5: CLASS_BUS,
}


class YOLODetector(BaseDetector):
    """
    YOLO-based detector supporting plate and vehicle models.
    Use class_offset=0 for plate models, or pass class_map for vehicle mapping.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        class_offset: int = 0,
        class_map: Optional[Dict[int, int]] = None,
    ):
        settings = get_settings()
        self.confidence = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.min_detection_confidence
        )
        self.device = "cpu"
        self.class_offset = class_offset
        self.class_map = class_map
        resolved_path = model_path or settings.plate_model_path

        self.logger = logger.bind(context=self.__class__.__name__)
        try:
            self.logger.info(f"Loading YOLO model from {resolved_path}...")
            self.model = YOLO(resolved_path)
            self.model.to(self.device)
            self.logger.success(f"YOLO model loaded successfully on {self.device}.")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence,
                verbose=False,
                device=self.device,
            )
            return _process_yolo_output(
                results,
                image,
                self.model,
                class_offset=self.class_offset,
                class_map=self.class_map,
            )
        except Exception as e:
            self.logger.error(f"YOLO inference error: {e}")
            return []


class ParallelDetector(BaseDetector):
    """
    Runs multiple detectors in parallel and merges their results.
    """

    def __init__(self, detectors: List[BaseDetector]):
        self.detectors = detectors
        self._executor = ThreadPoolExecutor(max_workers=len(detectors))
        self.logger = logger.bind(context=self.__class__.__name__)

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        futures = [self._executor.submit(d.detect, image) for d in self.detectors]

        merged: List[DetectionResult] = []
        for future in futures:
            try:
                merged.extend(future.result())
            except Exception as e:
                self.logger.error(f"Parallel detector error: {e}")

        merged.sort(key=lambda d: d.confidence, reverse=True)
        return merged


def _process_yolo_output(
    results,
    image: np.ndarray,
    model,
    *,
    class_offset: int = 0,
    class_map: Optional[Dict[int, int]] = None,
) -> List[DetectionResult]:
    detections: List[DetectionResult] = []
    img_h, img_w = image.shape[:2]

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords

            conf = float(box.conf[0])
            raw_cls = int(box.cls[0])

            if class_map is not None:
                cls = class_map.get(raw_cls)
                if cls is None:
                    continue
            else:
                cls = raw_cls + class_offset

            label = model.names[raw_cls]

            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(img_w, int(x2)), min(img_h, int(y2))

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

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections
