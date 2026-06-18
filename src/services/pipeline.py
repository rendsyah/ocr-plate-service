import time
from typing import Optional

import cv2
import numpy as np

from src.config import get_settings
from src.domain import BaseDetector, BaseNormalizer, BaseOCR, PlatePrediction
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

    def _preprocess(
        self, image: np.ndarray, debug_filename: Optional[str] = None
    ) -> np.ndarray:
        """
        Enhance the cropped plate image for better OCR accuracy.

        Uses adaptive upscaling based on the crop height to balance
        OCR accuracy and inference speed. Hard cropping is removed
        to better support tilted or slightly rotated license plates.
        """

        h, w = image.shape[:2]

        # Determine the appropriate upscale factor based on plate height.
        #
        # Very small plates need more enlargement to improve character
        # readability, while larger plates should avoid unnecessary
        # upscaling to keep OCR inference fast.
        if h < 32:
            scale_factor = 2.0
        elif h < 56:
            scale_factor = 1.5
        elif h < 72:
            scale_factor = 1.25
        else:
            scale_factor = 1.0

        # Upscale while preserving the original aspect ratio.
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

        # Debug save
        if debug_filename:
            self.storage.save_debug_image(
                upscaled,
                "preprocess",
                f"proc_{debug_filename}",
            )

        return upscaled

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

            best_det = detections[0]
            if best_det.image is None:
                self.logger.info("Best detection has no crop image.")
                return None

            # 2. Preprocessing
            start = time.time()
            # Generate a timestamp-based name if we need debug crops
            debug_name = (
                f"{int(time.time() * 1000)}.jpg"
                if self.settings.app_env == "development"
                else None
            )
            plate_img = self._preprocess(best_det.image, debug_name)
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
            # Rounding metrics for production-grade output
            metrics = {k: round(v, 2) for k, v in metrics.items()}
            metrics["total_ms"] = round((time.time() - overall_start) * 1000, 2)

            prediction = PlatePrediction(
                plate_number=norm_plate,
                vehicle_type=best_det.label,
                confidence=round((best_det.confidence + ocr_res.confidence) / 2, 4),
                detection_conf=round(best_det.confidence, 4),
                ocr_conf=round(ocr_res.confidence, 4),
                is_valid=is_valid,
                box=best_det.box,
                metadata={"metrics": metrics},
            )

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
