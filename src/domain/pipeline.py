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
    SPECIAL_PATTERN: str = r"^(RI|CD|CC|DF)[0-9]{1,8}$"

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

    # Tax date regex patterns
    # Tax: 4-digit year (2025) or starts with 0 (025)
    # fmt: off
    _INLINE_TAX = re.compile(r"^([A-Z]{1,2})(\d{1,4})(\d{4}|0\d{2,3})([A-Z0-9]{1,3})$")
    _TAIL_TAX = re.compile(r"^([A-Z]{1,2}\d{1,4}[A-Z]{1,3})(\d{3,4})$")
    # fmt: on

    # Soft swaps (commonly confused by OCR, very low penalty)
    # fmt: off
    SOFT_SWAPS = {
        ("I", "1"), ("1", "I"), ("O", "0"), ("0", "O"), ("B", "8"), ("8", "B"),
        ("S", "5"), ("5", "S"), ("Z", "2"), ("2", "Z"), ("G", "6"), ("6", "G"),
        ("T", "7"), ("7", "T"),
    }
    # fmt: on

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

        for clean in candidates_to_test:
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

                    score = self._score_candidate(
                        candidate,
                        clean,
                        prefix,
                        number,
                        suffix,
                        p_raw,
                        n_raw,
                        s_raw,
                        p_len,
                        n_len,
                        s_len,
                    )

                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
                        self.logger.debug(f"New best: {candidate} score={score:.1f}")

        return best_candidate

    def is_valid(self, plate: str) -> bool:
        if not plate:
            return False
        return bool(re.fullmatch(self.PLATE_PATTERN, plate)) or bool(
            re.fullmatch(self.SPECIAL_PATTERN, plate)
        )

    def _strip_tax_date(self, text: str) -> str:
        """
        Attempt to remove tax date noise (e.g., 'B12340125ABC' -> 'B1234ABC').
        Uses two patterns (inline + tail) with is_valid() as safety gate.
        """
        # Tail: B1234ABC0525 → B1234ABC (valid?)
        m = self._TAIL_TAX.match(text)
        if m:
            stripped = m.group(1)
            if self.is_valid(stripped):
                return stripped
        # Inline: B12025ABC → B12ABC or B12342025ABC → B1234ABC
        # Tax must be 4-digit (year) or start with 0 (regional code).
        # Suffix allows digits (mapped via LETTER_MAP) to handle OCR errors
        # like 5→S, 8→B in the suffix portion.
        m = self._INLINE_TAX.match(text)
        if m:
            number, tax = m.group(2), m.group(3)
            if len(number) >= len(tax) or tax.startswith("0"):
                # Total digit count guard: if number + tax ≤ 4 digits, the "tax"
                # is likely part of the plate number (e.g., B5046T8T where
                # number=5, tax=046 → 1+3=4 → ambiguous, skip).
                if len(number) + len(tax) <= 4:
                    return text
                suffix_raw = m.group(4)
                # Suffix must contain at least one actual letter (not just mapped
                # from digits). This prevents false splits like RI12340525 → RI1234S
                # where the entire "suffix" is actually part of the registration number.
                if not any(c.isalpha() for c in suffix_raw):
                    return text
                suffix = "".join(self.LETTER_MAP.get(c, c) for c in suffix_raw)
                stripped = f"{m.group(1)}{number}{suffix}"
                if self.is_valid(stripped):
                    return stripped
        return text

    def _score_candidate(
        self,
        candidate: str,
        clean: str,
        prefix: str,
        number: str,
        suffix: str,
        p_raw: str,
        n_raw: str,
        s_raw: str,
        p_len: int,
        n_len: int,
        s_len: int,
    ) -> float:
        is_special = prefix in self.SPECIAL_PREFIXES
        score = 50.0

        # 1. Hardware Integrity (Changes Analysis)
        hard_changes = 0
        soft_changes = 0
        for orig, fix in zip(clean, candidate):
            if orig != fix:
                if (orig, fix) in self.SOFT_SWAPS:
                    soft_changes += 1
                else:
                    hard_changes += 1

        if hard_changes > 1 or (hard_changes + soft_changes) / len(candidate) > 0.4:
            return -999.0

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

        # 4. Tax Date Protection
        if not is_special and s_len > 0 and s_raw.isdigit():
            score -= 25.0

        # 4b. Digit→letter mapping in suffix
        if s_len > 0:
            mapped_digits = sum(
                1 for oc, sc in zip(s_raw, suffix) if oc.isdigit() and sc.isalpha()
            )
            if mapped_digits > 0:
                score -= 30.0

        # 5. Length Reward
        score += len(candidate) * 5.0

        # 6. Originality Bonus
        if p_raw[0] == prefix[0]:
            score += 10.0

        return score
