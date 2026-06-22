import time
from typing import Optional

import numpy as np

from src.config import get_settings
from src.domain import BaseOCR, OCRResult
from src.utils import logger


class PaddleOCR(BaseOCR):
    """
    PaddleOCR implementation for License Plate Recognition.
    Compatible with PaddleOCR v3.7+ (PaddleX-based).
    """

    _ALLOWLIST = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(self):
        self.settings = get_settings()
        self.logger = logger.bind(context=self.__class__.__name__)

        try:
            from paddleocr import PaddleOCR as PaddleEngine

            device = "cpu"
            self.engine = PaddleEngine(
                lang="en",
                text_detection_model_name=self.settings.ocr_detection_model,
                text_recognition_model_name=self.settings.ocr_recognition_model,
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
            self.logger.success(f"PaddleOCR initialized successfully on {device}.")
        except ImportError:
            self.logger.error("paddleocr not installed.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def recognize(self, image: np.ndarray) -> Optional[OCRResult]:
        """Recognize text using PaddleOCR from a plate crop."""
        if image is None or image.size == 0:
            return None

        start_time = time.time()
        try:
            # v3.7+ returns a list of dictionaries per image
            results = self.engine.predict(image)

            if not results or not isinstance(results[0], dict):
                return None

            res = results[0]
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("rec_polys", [])

            if not texts:
                return None

            items = []
            for text, score, poly in zip(texts, scores, polys):
                if not text:
                    continue

                poly_arr = np.array(poly)
                x_min = float(poly_arr[:, 0].min())
                y_min = float(poly_arr[:, 1].min())
                y_max = float(poly_arr[:, 1].max())
                y_center = (y_min + y_max) / 2

                items.append(
                    {
                        "text": str(text).upper().replace(" ", ""),
                        "score": float(score),
                        "x": x_min,
                        "y": y_center,
                        "height": y_max - y_min,
                    }
                )

            if not items:
                return None

            # 1. Find the "Anchor" segment (largest height, likely the main plate number)
            # We sort by height descending to pick the most prominent text segment.
            items.sort(key=lambda i: i["height"], reverse=True)
            anchor = items[0]

            # 2. Filter: Keep segments that are on the same line as the anchor
            # AND have a similar height. This effectively removes tax dates (bottom row)
            # and small noise while allowing for tilted plates.
            main_items = []
            threshold = anchor["height"] * 0.5
            for i in items:
                y_dist = abs(i["y"] - anchor["y"])
                h_ratio = i["height"] / anchor["height"]

                if y_dist < threshold and h_ratio > 0.5:
                    main_items.append(i)

            # Fallback if filtering was too aggressive
            if not main_items:
                main_items = items

            # 3. Sort primarily by X (left-to-right)
            main_items.sort(key=lambda i: i["x"])

            # 4. Remove trailing numeric-only segments that look like tax dates
            # Uses actual segment boundaries from the detection model, more
            # accurate than regex guessing. Handles e.g. "B1234ABC"+"0525".
            while (
                len(main_items) > 1
                and main_items[-1]["text"].isdigit()
                and 2 <= len(main_items[-1]["text"]) <= 4
            ):
                remaining = "".join(i["text"] for i in main_items[:-1])
                if (
                    remaining
                    and any(c.isalpha() for c in remaining)
                    and any(c.isdigit() for c in remaining)
                ):
                    self.logger.debug(
                        f"Removed trailing tax segment: {main_items[-1]['text']}"
                    )
                    main_items.pop()
                else:
                    break

            # Clean and combine text
            raw_text = "".join(i["text"] for i in main_items)
            clean_text = "".join(c for c in raw_text if c in self._ALLOWLIST)

            if not clean_text:
                return None

            # Standardized to mean confidence
            confidence = sum(i["score"] for i in main_items) / len(main_items)

            latency = (time.time() - start_time) * 1000
            self.logger.debug(f"PaddleOCR inference: {clean_text} | {latency:.1f}ms")

            return OCRResult(text=clean_text, confidence=confidence)

        except Exception as e:
            self.logger.warning(f"PaddleOCR recognition error: {e}")
            return None
