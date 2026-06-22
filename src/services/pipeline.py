import math
import time
from typing import List, Optional

import cv2
import numpy as np

from src.config import get_settings
from src.domain import (
    BaseDetector,
    BaseNormalizer,
    BaseOCR,
    DetectionResult,
    PlatePrediction,
)
from src.services.detector import CLASS_LICENSE_PLATE, VEHICLE_CLASS_IDS
from src.services.storage import StorageService
from src.utils import logger


class OCRPipeline:
    """
    End-to-end pipeline: Detect → Preprocess → OCR → Normalize → Validate.
    Optimized for low-latency production environments with telemetry and debug hooks.
    """

    def __init__(
        self,
        detector: BaseDetector,
        ocr: BaseOCR,
        normalizer: BaseNormalizer,
        storage: StorageService,
    ):
        self.detector = detector
        self.ocr = ocr
        self.normalizer = normalizer
        self.storage = storage
        self.settings = get_settings()
        self.logger = logger.bind(context=self.__class__.__name__)

    def process_frame(self, image: np.ndarray) -> Optional[PlatePrediction]:
        """
        Process a single frame through the full OCR pipeline with latency tracking.
        """
        if image is None or image.size == 0:
            return None

        metrics = {}
        overall_start = time.time()

        try:
            # 1. Detection
            start = time.time()
            detections = self.detector.detect(image)
            metrics["detection_ms"] = (time.time() - start) * 1000

            if not detections:
                return None

            plates = [d for d in detections if d.class_id == CLASS_LICENSE_PLATE]
            if not plates:
                self.logger.info("No license plate detected.")
                return None

            best_plate = plates[0]
            if best_plate.image is None:
                self.logger.info("Best plate detection has no crop image.")
                return None

            vehicle_type = self._find_vehicle_type(best_plate.box, detections)

            # 2. Preprocessing
            start = time.time()
            plate_img = self._preprocess(best_plate.image)
            metrics["preprocess_ms"] = (time.time() - start) * 1000

            # 3. OCR Recognition
            start = time.time()
            ocr_res = self.ocr.recognize(plate_img)
            metrics["ocr_ms"] = (time.time() - start) * 1000

            if not ocr_res:
                self.logger.info("OCR returned no result.")
                return None

            if ocr_res.confidence < self.settings.min_ocr_confidence:
                self.logger.bind(
                    conf=round(ocr_res.confidence, 4),
                    min_conf=self.settings.min_ocr_confidence,
                ).info("OCR confidence too low")
                return None

            # 4. Normalization & Validation
            start = time.time()
            norm_plate = self.normalizer.normalize(ocr_res.text)
            is_valid = self.normalizer.is_valid(norm_plate)
            metrics["normalize_ms"] = (time.time() - start) * 1000

            # 5. Final Metrics & Output
            metrics = {k: round(v, 2) for k, v in metrics.items()}
            metrics["total_ms"] = round((time.time() - overall_start) * 1000, 2)

            prediction = PlatePrediction(
                plate_number=norm_plate,
                vehicle_type=vehicle_type,
                confidence=round((best_plate.confidence + ocr_res.confidence) / 2, 4),
                detection_conf=round(best_plate.confidence, 4),
                ocr_conf=round(ocr_res.confidence, 4),
                is_valid=is_valid,
                box=best_plate.box,
                metadata={"metrics": metrics},
            )

            # Save results
            snapshot_filename, preprocess_filename = self.storage.save_results(
                image, prediction, plate_img
            )
            prediction.metadata["snapshot_filename"] = snapshot_filename
            prediction.metadata["preprocess_filename"] = preprocess_filename

            if is_valid:
                self.logger.bind(
                    plate=norm_plate,
                    raw=ocr_res.text,
                    conf=round(ocr_res.confidence, 4),
                    metrics=metrics,
                ).success(f"OCR Success: {norm_plate}")
            else:
                self.logger.bind(
                    plate=norm_plate,
                    raw=ocr_res.text,
                    conf=round(ocr_res.confidence, 4),
                    metrics=metrics,
                ).info(f"Invalid Format: {norm_plate}")

            return prediction

        except Exception as e:
            self.logger.exception(f"Pipeline processing error: {e}")
            return None

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        if h < 32:
            scale_factor = 2.0
        elif h < 56:
            scale_factor = 1.5
        elif h < 72:
            scale_factor = 1.25
        else:
            scale_factor = 1.0

        if scale_factor != 1.0:
            up_h = int(h * scale_factor)
            up_w = int(w * scale_factor)

            upscaled = cv2.resize(
                image,
                (up_w, up_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            upscaled = image

        return upscaled

    @staticmethod
    def _find_vehicle_type(
        plate_box: List[float],
        detections: List[DetectionResult],
    ) -> str:
        plate_cx = (plate_box[0] + plate_box[2]) / 2
        plate_cy = (plate_box[1] + plate_box[3]) / 2

        best_vehicle: Optional[DetectionResult] = None
        best_dist = math.inf

        for det in detections:
            if det.class_id not in VEHICLE_CLASS_IDS:
                continue

            vbox = det.box
            if vbox[0] <= plate_cx <= vbox[2] and vbox[1] <= plate_cy <= vbox[3]:
                return det.label

            vcx = (vbox[0] + vbox[2]) / 2
            vcy = (vbox[1] + vbox[3]) / 2
            dist = math.hypot(plate_cx - vcx, plate_cy - vcy)
            if dist < best_dist:
                best_dist = dist
                best_vehicle = det

        return best_vehicle.label if best_vehicle else "unknown"
