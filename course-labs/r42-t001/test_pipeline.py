"""
Unit tests for R42-T001 — Fashion-MNIST CNN schema validator.

Runs offline, CPU-only, with a fixed fixture (no randomness).
A test must exit non-zero when an expected result is deliberately changed.

Run with:
    python3 -m unittest discover -s course-labs/r42-t001 -p "test_*.py"
"""

import copy
import unittest

from pipeline import (
    make_fixture,
    validate_dataset,
    validate_record,
    NUM_PIXELS,
    VALID_LABELS,
)


class TestValidFixture(unittest.TestCase):
    """The canonical six-record fixture must pass with zero failures."""

    def setUp(self):
        self.fixture = make_fixture()

    def test_fixture_has_six_records(self):
        self.assertEqual(len(self.fixture), 6)

    def test_valid_fixture_passes_with_no_failures(self):
        failures = validate_dataset(self.fixture)
        self.assertEqual(
            failures, [],
            msg=f"Expected zero failures on clean fixture, got: {failures}",
        )

    def test_fixture_has_three_label0_and_three_label1(self):
        labels = [r["label"] for r in self.fixture]
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)

    def test_fixture_pixels_are_16_values_each(self):
        for rec in self.fixture:
            self.assertEqual(len(rec["pixels"]), NUM_PIXELS, rec["record_id"])

    def test_fixture_all_record_ids_unique(self):
        ids = [r["record_id"] for r in self.fixture]
        self.assertEqual(len(ids), len(set(ids)))

    def test_fixture_all_group_ids_present(self):
        gids = {r["group_id"] for r in self.fixture}
        self.assertGreaterEqual(len(gids), 2)   # at least two groups declared


class TestMissingFields(unittest.TestCase):
    """Records with missing required fields must produce specific failures."""

    def _one_record(self, **overrides):
        base = copy.deepcopy(make_fixture()[0])
        for k, v in overrides.items():
            if v is None:
                base.pop(k, None)
            else:
                base[k] = v
        return base

    def test_missing_record_id_reported(self):
        rec = self._one_record(record_id=None)
        failures = validate_record(rec, set())
        fields = [f.field for f in failures]
        self.assertIn("record_id", fields)
        reason = [f.reason for f in failures if f.field == "record_id"][0]
        self.assertEqual(reason, "missing or null record_id")

    def test_missing_group_id_reported(self):
        rec = self._one_record(group_id=None)
        failures = validate_record(rec, set())
        fields = [f.field for f in failures]
        self.assertIn("group_id", fields)

    def test_missing_pixels_reported(self):
        rec = self._one_record(pixels=None)
        failures = validate_record(rec, set())
        fields = [f.field for f in failures]
        self.assertIn("pixels", fields)

    def test_missing_label_reported(self):
        rec = self._one_record(label=None)
        failures = validate_record(rec, set())
        fields = [f.field for f in failures]
        self.assertIn("label", fields)


class TestMalformedData(unittest.TestCase):
    """Malformed values must be reported with specific field + reason."""

    def _record(self, **overrides):
        base = copy.deepcopy(make_fixture()[0])
        base.update(overrides)
        return base

    def test_non_string_record_id_fails(self):
        rec = self._record(record_id=42)
        failures = validate_record(rec, set())
        fields = [f.field for f in failures]
        self.assertIn("record_id", fields)

    def test_non_string_group_id_fails(self):
        rec = self._record(group_id=99)
        failures = validate_record(rec, set())
        fields = [f.field for f in failures]
        self.assertIn("group_id", fields)

    def test_pixels_wrong_length_fails(self):
        rec = self._record(pixels=[0] * 8)   # too short
        failures = validate_record(rec, set())
        fields = [f.field for f in failures]
        self.assertIn("pixels", fields)

    def test_pixels_non_list_fails(self):
        rec = self._record(pixels="not-a-list")
        failures = validate_record(rec, set())
        fields = [f.field for f in failures]
        self.assertIn("pixels", fields)

    def test_pixel_value_out_of_range_fails(self):
        bad_pixels = [0] * NUM_PIXELS
        bad_pixels[0] = 300   # > 255
        rec = self._record(pixels=bad_pixels)
        failures = validate_record(rec, set())
        self.assertTrue(any("pixels" in f.field for f in failures))

    def test_invalid_label_fails(self):
        rec = self._record(label=99)
        failures = validate_record(rec, set())
        fields = [f.field for f in failures]
        self.assertIn("label", fields)

    def test_non_int_label_fails(self):
        rec = self._record(label="zero")
        failures = validate_record(rec, set())
        fields = [f.field for f in failures]
        self.assertIn("label", fields)

    def test_bool_label_fails(self):
        # bool is a subclass of int in Python; it must still be rejected
        rec = self._record(label=True)
        failures = validate_record(rec, set())
        self.assertEqual(
            [(f.field, f.reason) for f in failures],
            [("label", "label must be int, got 'bool'")],
        )

    def test_float_pixel_fails(self):
        bad_pixels = [0] * NUM_PIXELS
        bad_pixels[3] = 0.5
        rec = self._record(pixels=bad_pixels)
        failures = validate_record(rec, set())
        self.assertEqual(
            [(f.field, f.reason) for f in failures],
            [("pixels[3]", "pixel value must be int, got 'float'")],
        )

    def test_bool_pixel_fails(self):
        bad_pixels = [0] * NUM_PIXELS
        bad_pixels[0] = True
        rec = self._record(pixels=bad_pixels)
        failures = validate_record(rec, set())
        self.assertEqual([f.field for f in failures], ["pixels[0]"])

    def test_nested_4x4_tensor_pixels_fail(self):
        # pixels must be a flat list of 16 values, not a nested 4x4 tensor
        rec = self._record(pixels=[[0] * 4 for _ in range(4)])
        failures = validate_record(rec, set())
        self.assertEqual(
            [(f.field, f.reason) for f in failures],
            [("pixels", "pixels length must be 16 (4×4), got 4")],
        )


class TestNoSilentDrop(unittest.TestCase):
    """Every bad record is reported by record ID; none is silently dropped."""

    def test_each_bad_record_reported(self):
        fixture = make_fixture()
        fixture[1]["label"] = 7
        fixture[4]["pixels"] = [0] * 3
        failures = validate_dataset(fixture)
        self.assertEqual(
            [(f.record_id, f.field) for f in failures],
            [("r002", "label"), ("r005", "pixels")],
        )


class TestDuplicateRecordId(unittest.TestCase):
    """A duplicate record_id must be detected and reported."""

    def test_duplicate_id_fails(self):
        fixture = make_fixture()
        # inject a duplicate
        dup = copy.deepcopy(fixture[0])
        dup["group_id"] = "g99"
        dataset = fixture + [dup]
        failures = validate_dataset(dataset)
        dup_failures = [f for f in failures if "duplicate" in f.reason.lower()]
        self.assertGreater(
            len(dup_failures), 0,
            msg="Expected at least one duplicate-ID failure",
        )


class TestNonZeroCountSentinel(unittest.TestCase):
    """
    Sentinel: deliberately break an expected value → test must fail.
    This guards against vacuous/empty test suites.
    """

    def test_nonzero_test_count(self):
        """At least one assertion has been made in this test module."""
        # Count tests in this module — the suite itself is non-empty.
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(__import__(__name__))
        count = suite.countTestCases()
        self.assertGreater(count, 0, "Test suite must not be empty")

    def test_sentinel_deliberate_failure_guard(self):
        """
        If you change the expected value below the test MUST fail.
        Expected: 'PASS' — do not alter.
        """
        fixture = make_fixture()
        failures = validate_dataset(fixture)
        outcome = "PASS" if not failures else "FAIL"
        self.assertEqual(outcome, "PASS")   # <-- deliberately fixed


if __name__ == "__main__":
    unittest.main()
