import random
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from src.api.v1.schemas import OCRData, OCRMetadata, OCRMetrics
from src.config import get_settings
from src.services import OCRPipeline
from src.utils import logger


class OCRApiService:
    """
    API-specific service for OCR operations.
    Handles business logic and failure states before returning to controllers.
    """

    def __init__(self, pipeline: OCRPipeline):
        self.pipeline = pipeline
        self.settings = get_settings()
        self.logger = logger.bind(context=self.__class__.__name__)

    async def predict(self, file: UploadFile) -> OCRData:
        """
        OCR prediction from an uploaded image.
        """
        image_bytes = await file.read()
        upload_filename = file.filename
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode uploaded image: {upload_filename}",
            )

        prediction = self.pipeline.process_frame(img)

        if not prediction:
            raise HTTPException(
                status_code=422, detail="No license plate detected in uploaded image"
            )

        return self._build_ocr_data(prediction, upload_filename=upload_filename)

    async def predict_test(self) -> OCRData:
        """
        Logic for picking a random sample and processing it.
        Raises HTTPException if any step fails.
        """
        sample_dir = Path("tests/samples")
        if not sample_dir.exists():
            raise HTTPException(status_code=404, detail="Sample directory not found")

        image_extensions = (".jpg", ".jpeg", ".png", ".webp")
        samples = [
            p for p in sample_dir.glob("**/*") if p.suffix.lower() in image_extensions
        ]
        if not samples:
            raise HTTPException(status_code=404, detail="No sample images found")

        sample_path = random.choice(samples)
        img = cv2.imread(str(sample_path))
        if img is None:
            raise HTTPException(
                status_code=500, detail=f"Failed to read sample: {sample_path.name}"
            )

        prediction = self.pipeline.process_frame(img)

        if not prediction:
            raise HTTPException(
                status_code=422, detail="No license plate detected in sample"
            )

        return self._build_ocr_data(prediction, sample_name=sample_path.name)

    @staticmethod
    def _build_ocr_data(prediction, upload_filename=None, sample_name=None):
        return OCRData(
            plate_number=prediction.plate_number,
            vehicle_type=prediction.vehicle_type,
            confidence=prediction.confidence,
            ocr_conf=prediction.ocr_conf,
            detection_conf=prediction.detection_conf,
            is_valid=prediction.is_valid,
            snapshot_filename=prediction.metadata.get("snapshot_filename"),
            preprocess_filename=prediction.metadata.get("preprocess_filename"),
            sample_used=upload_filename or sample_name,
            metadata=OCRMetadata(metrics=OCRMetrics(**prediction.metadata["metrics"])),
        )
