import re
from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from src.utils import logger


class PlatePrediction(BaseModel):
    """Represents the final prediction result from the OCR pipeline."""

    plate_number: str
    vehicle_type: str
    confidence: float  # Combined confidence (detection + ocr average)
    detection_conf: float
    ocr_conf: float
    is_valid: bool
    box: List[float]  # [x1, y1, x2, y2]
    metadata: dict = {}  # Latency and other debug info


class BaseNormalizer(ABC):
    """Base class for country-specific plate normalizers."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize OCR result."""
        pass

    @abstractmethod
    def is_valid(self, plate: str) -> bool:
        """Validate plate format."""
        pass


class IndonesianPlateNormalizer(BaseNormalizer):
    """
    Handles Indonesian-specific plate character correction and validation.
    Production-grade logic: Balanced between flexibility and integrity.
    """

    def __init__(self):
        self.logger = logger.bind(context=self.__class__.__name__)

    # Primary Pattern: [1-2 letters][1-4 digits][1-3 letters]
    PLATE_PATTERN: str = r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$"

    # Special Pattern: [RI/CD/CC/DF][1-4 digits] (No suffix)
    SPECIAL_PATTERN: str = r"^(RI|CD|CC|DF)[0-9]{1,4}$"

    # Valid Indonesian regional codes
    # fmt: off
    VALID_PREFIXES = {
        "A", "B", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "Z",
        "AB", "AE", "AG", "BA", "BB", "BD", "BE", "BG", "BH", "BK", "BL", "BM", "BN", "BP",
        "DA", "DB", "DC", "DD", "DE", "DG", "DH", "DK", "DL", "DM", "DN", "DT",
        "EA", "EB", "ED", "EF", "KB", "KH", "KT", "KU", "PA", "PB",
    }
    SPECIAL_PREFIXES = {"RI", "CD", "CC", "DF"}
    # fmt: on

    # Digit map for common OCR errors
    # fmt: off
    DIGIT_MAP: dict[str, str] = {
        "O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7",
    }
    # fmt: on

    # Letter map for common OCR errors
    # fmt: off
    LETTER_MAP: dict[str, str] = {
        "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T",
    }
    # fmt: on

    # Soft swaps (commonly confused by OCR, very low penalty)
    # fmt: off
    SOFT_SWAPS = {
        ("I", "1"), ("1", "I"), ("O", "0"), ("0", "O"), ("B", "8"), ("8", "B"),
        ("S", "5"), ("5", "S"), ("Z", "2"), ("2", "Z"), ("G", "6"), ("6", "G"),
        ("T", "7"), ("7", "T"),
    }
    # fmt: on

    def _strip_tax_date(self, text: str) -> str:
        """
        Attempt to remove tax date noise (e.g., 'B12340125ABC' -> 'B1234ABC').
        Requires at least 2 digits for the main number to avoid over-stripping beautiful plates.
        """
        # Match a pattern like [letters][2-4 digits][2-4 noise digits][1-3 letters]
        match = re.search(r"([A-Z]{1,2})([0-9]{2,4})([0-9]{2,4})([A-Z]{1,3})", text)
        if match:
            return f"{match.group(1)}{match.group(2)}{match.group(4)}"
        return text

    def normalize(self, text: str) -> str:
        """
        Normalize OCR result into Indonesian plate format.
        Uses a weighted scoring system to ensure 'smart' but 'not forced' results.
        """
        if not text:
            return ""

        # Pre-clean: Remove non-alphanumeric
        raw_text = re.sub(r"[^A-Z0-9]", "", text.upper().replace(" ", ""))
        if not raw_text:
            return ""

        # Create candidates: 1. Original 2. Tax-stripped version
        candidates_to_test = {raw_text, self._strip_tax_date(raw_text)}

        best_candidate = raw_text
        best_score = -500.0

        for raw_clean in candidates_to_test:
            length_full = len(raw_clean)
            for sub_start in range(length_full):
                # Window from length 2 (RI 1) to 9 (AB 1234 ABC)
                for sub_end in range(
                    sub_start + 2, min(sub_start + 10, length_full + 1)
                ):
                    clean = raw_clean[sub_start:sub_end]
                    length = len(clean)

                    for p_len in (1, 2):
                        for n_len in (1, 2, 3, 4):
                            s_len = length - p_len - n_len
                            if s_len < 0 or s_len > 3:
                                continue

                            p_raw = clean[:p_len]
                            n_raw = clean[p_len : p_len + n_len]
                            s_raw = clean[p_len + n_len :]

                            prefix = "".join(self.LETTER_MAP.get(c, c) for c in p_raw)
                            number = "".join(self.DIGIT_MAP.get(c, c) for c in n_raw)
                            suffix = "".join(self.LETTER_MAP.get(c, c) for c in s_raw)

                            candidate = f"{prefix}{number}{suffix}"

                            if not self.is_valid(candidate):
                                continue

                            # --- SMART SCORING ---
                            score = 50.0
                            is_special = prefix in self.SPECIAL_PREFIXES

                            # 1. Hardware Integrity (Changes Analysis)
                            hard_changes = 0
                            soft_changes = 0
                            for orig, fix in zip(clean, candidate):
                                if orig != fix:
                                    if (orig, fix) in self.SOFT_SWAPS:
                                        soft_changes += 1
                                    else:
                                        hard_changes += 1

                            if (
                                hard_changes > 1
                                or (hard_changes + soft_changes) / length > 0.4
                            ):
                                continue

                            score -= hard_changes * 15.0
                            score -= soft_changes * 2.0

                            # 2. Prefix Awareness
                            if prefix in self.VALID_PREFIXES or is_special:
                                score += 35.0
                            else:
                                score -= 15.0

                            # 3. Numeric Block Sanity
                            if number.startswith("0") and len(number) > 1:
                                score -= 15.0

                            # 4. Tax Date Protection (Tie-breaker for standard plates)
                            # If standard plate suffix was originally all digits, it's likely a tax date
                            if not is_special and s_len > 0 and s_raw.isdigit():
                                score -= 25.0

                            # 5. Length Reward
                            score += length * 8.0

                            # 6. Originality Bonus
                            if p_raw[0] == prefix[0]:
                                score += 10.0

                            if score > best_score:
                                best_score = score
                                best_candidate = candidate
                                self.logger.debug(
                                    f"New best: {candidate} score={score:.1f}"
                                )

        return best_candidate

    def is_valid(self, plate: str) -> bool:
        if not plate:
            return False
        return bool(re.fullmatch(self.PLATE_PATTERN, plate)) or bool(
            re.fullmatch(self.SPECIAL_PATTERN, plate)
        )
