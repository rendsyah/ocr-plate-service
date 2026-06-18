import os
import time
from pathlib import Path

import cv2

from src.config import get_settings
from src.services import (
    OCRPipeline,
    get_detector,
    get_normalizer,
    get_ocr,
    get_storage,
)
from src.utils import logger, setup_app_logging

# Constants for local validation
SAMPLE_DIR = "tests/samples"


def main():
    """
    OCR Local Validation Tool.
    Runs the pipeline against images in tests/samples and saves results via StorageService.
    """
    settings = get_settings()

    # Initialize logging (suppresses warnings globally)
    setup_app_logging(log_level=settings.log_level)

    # Ensure directories exist
    os.makedirs(settings.snapshot_dir, exist_ok=True)
    if not os.path.exists(SAMPLE_DIR):
        logger.warning(f"Sample directory not found: {SAMPLE_DIR}")
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        return

    # 1. Initialize Engines
    try:
        detector = get_detector()
        ocr = get_ocr()
        normalizer = get_normalizer()
        storage = get_storage()
        pipeline = OCRPipeline(detector, ocr, normalizer, storage)

        logger.info(f"Engines initialized (Engine: {settings.ocr_engine})")
    except Exception as e:
        logger.critical(f"Failed to initialize engines: {e}")
        return

    # 2. Get Images
    image_extensions = (".jpg", ".jpeg", ".png", ".webp")
    image_paths = [
        p for p in Path(SAMPLE_DIR).glob("**/*") if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        logger.warning(f"No images found in {SAMPLE_DIR}. Please add images to test.")
        return

    logger.info(f"Starting validation on {len(image_paths)} images...")

    results = []
    total_start_time = time.time()

    for img_path in image_paths:
        logger.info(f"Processing: {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Could not read image: {img_path}")
            continue

        # Run Pipeline
        start_time = time.time()
        prediction = pipeline.process_frame(img)
        end_time = time.time()

        proc_time = (end_time - start_time) * 1000

        if prediction:
            status = "VALID" if prediction.is_valid else "INVALID"
            metrics = prediction.metadata.get("metrics", {})
            logger.info(
                f"Result: {prediction.plate_number} | {status} | "
                f"Total: {metrics.get('total_ms', 0):.1f}ms"
            )
            results.append(
                {
                    "file": img_path.name,
                    "plate": prediction.plate_number,
                    "valid": prediction.is_valid,
                    "conf": prediction.confidence,
                    "time": proc_time,
                }
            )

            # Save visual result using StorageService for consistency
            storage.save_snapshot(img, prediction)
        else:
            logger.warning(f"No plate detected in {img_path.name}")

    total_time = time.time() - total_start_time

    # 3. Summary Report
    logger.info("=" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info(f"Total Images: {len(image_paths)}")
    logger.info(f"Plates Found: {len(results)}")
    if results:
        avg_time = sum(r["time"] for r in results) / len(results)
        valid_count = sum(1 for r in results if r["valid"])
        logger.info(f"Valid Patterns: {valid_count}/{len(results)}")
        logger.info(f"Average Speed: {avg_time:.2f} ms/frame")
    logger.info(f"Total Duration: {total_time:.2f} seconds")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
