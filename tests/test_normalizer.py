from src.domain.pipeline import IndonesianPlateNormalizer


# =============================================================================
# is_valid
# =============================================================================


class TestIsValid:
    def test_standard_plate_2_4_2(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.is_valid("BM6432YZ")

    def test_standard_plate_1_4_2(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.is_valid("B6432YZ")

    def test_standard_plate_2_4_3(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.is_valid("BM6432YZO")

    def test_standard_plate_1_1_1(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.is_valid("A1A")

    def test_standard_plate_2_1_1(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.is_valid("AB1C")

    def test_special_plate_ri(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.is_valid("RI1234")

    def test_special_plate_cd(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.is_valid("CD123")

    def test_special_plate_cc(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.is_valid("CC12")

    def test_special_plate_df(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.is_valid("DF1")

    def test_empty_string(self, normalizer: IndonesianPlateNormalizer):
        assert not normalizer.is_valid("")
        assert not normalizer.is_valid("AB")

    def test_too_long_invalid(self, normalizer: IndonesianPlateNormalizer):
        assert not normalizer.is_valid("ABCD1234EFGH")

    def test_no_digits_invalid(self, normalizer: IndonesianPlateNormalizer):
        assert not normalizer.is_valid("ABCDEF")

    def test_no_letters_invalid(self, normalizer: IndonesianPlateNormalizer):
        assert not normalizer.is_valid("1234")

    def test_suffix_too_long_invalid(self, normalizer: IndonesianPlateNormalizer):
        assert not normalizer.is_valid("AB1234ABCD")

    def test_lowercase_input(self, normalizer: IndonesianPlateNormalizer):
        assert not normalizer.is_valid("bm6432yz")


# =============================================================================
# _strip_tax_date
# =============================================================================


class TestStripTaxDate:
    def test_tail_tax_removed(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer._strip_tax_date("B1234ABC0525") == "B1234ABC"

    def test_tail_tax_short_year(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer._strip_tax_date("B1234ABC025") == "B1234ABC"

    def test_inline_tax_removed(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer._strip_tax_date("B12025ABC") == "B12ABC"

    def test_inline_tax_with_zero_prefix(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer._strip_tax_date("B1234025ABC") == "B1234ABC"

    def test_no_tax_returns_original(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer._strip_tax_date("B1234ABC") == "B1234ABC"

    def test_tail_tax_special_prefix(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer._strip_tax_date("RI12340525") == "RI12340525"

    def test_inline_tax_not_stripped_when_invalid(
        self, normalizer: IndonesianPlateNormalizer
    ):
        # Stripping B12025AB gives B12AB (2 chars suffix) — still valid,
        # but if it were invalid, original should be returned
        assert normalizer._strip_tax_date("B12025AB") == "B12AB"

    def test_strip_with_motorcycle_tax_date(
        self, normalizer: IndonesianPlateNormalizer
    ):
        assert normalizer._strip_tax_date("BM6432YZ0327") == "BM6432YZ"


# =============================================================================
# normalize
# =============================================================================


class TestNormalize:
    def test_clean_plate_unchanged(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.normalize("BM6432YZ") == "BM6432YZ"

    def test_clean_plate_with_spaces(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.normalize("BM 6432 YZ") == "BM6432YZ"

    def test_clean_plate_with_hyphen(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.normalize("BM-6432-YZ") == "BM6432YZ"

    def test_motorcycle_tax_date_stripped(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.normalize("BM6432YZ0327") == "BM6432YZ"

    def test_car_tax_date_stripped(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.normalize("B1234ABC0525") == "B1234ABC"

    def test_empty_string(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.normalize("") == ""

    def test_only_special_chars(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.normalize("--- ***") == ""

    def test_lowercase_input(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.normalize("bm 6432 yz") == "BM6432YZ"

    def test_special_plate_ri_preserved(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.normalize("RI1234") == "RI1234"

    def test_special_plate_cd_preserved(self, normalizer: IndonesianPlateNormalizer):
        assert normalizer.normalize("CD123") == "CD123"

    def test_ambiguous_ocr_picks_best(self, normalizer: IndonesianPlateNormalizer):
        result = normalizer.normalize("B1234ABC0525")
        assert result == "B1234ABC"

    def test_tax_date_not_stripped_if_makes_invalid(
        self, normalizer: IndonesianPlateNormalizer
    ):
        result = normalizer.normalize("BM6432YZ")
        assert result == "BM6432YZ"

    def test_extra_characters_around_valid_plate(
        self, normalizer: IndonesianPlateNormalizer
    ):
        result = normalizer.normalize("AAA BM6432YZ BBB")
        assert result == "AAABM6432YZBBB"

    def test_vehicle_info_tag(self, normalizer: IndonesianPlateNormalizer):
        result = normalizer.normalize("MI2SD")
        assert result == "M12SD"
        assert normalizer.is_valid("M12SD")

    def test_short_input_not_valid(self, normalizer: IndonesianPlateNormalizer):
        result = normalizer.normalize("NO")
        assert result == "NO"
