"""Explicit review choices retain precedence through CLI default-keep merging."""
import copy
import unittest

from fpvsesh.cli import merge_reviewed_keeps
from fpvsesh.planner import plan
from tests.test_pacing import fixture


class ReviewedKeepPrecedenceTests(unittest.TestCase):
    def setUp(self):
        timeline, self.probes, _, _ = fixture()
        self.candidates = copy.deepcopy(timeline["shots"])
        self.candidates[0]["review_key"] = "exact-first"
        self.reviews = [{"key": "exact-first", "keep": True}]

    def test_explicit_exclude_overrides_review_default_keep(self):
        overrides = {"exclude": ["0"]}
        merged = merge_reviewed_keeps(overrides, self.candidates, self.reviews)
        self.assertEqual(merged["exclude"], ["0"])
        self.assertNotIn("0", merged["keep"])
        result = plan(self.candidates, self.probes, "60/1", overrides=merged)
        self.assertEqual([shot["id"] for shot in result["shots"]], ["1"])
        self.assertEqual(overrides, {"exclude": ["0"]})

    def test_existing_explicit_conflict_still_reaches_planner_validation(self):
        overrides = {"keep": ["0"], "exclude": ["0"]}
        merged = merge_reviewed_keeps(overrides, self.candidates, self.reviews)
        self.assertEqual(merged, overrides)
        with self.assertRaisesRegex(ValueError, "both kept and excluded"):
            plan(self.candidates, self.probes, "60/1", overrides=merged)

    def test_review_keep_is_added_once_without_losing_explicit_choices(self):
        overrides = {"keep": ["1"], "order": ["0", "1"]}
        merged = merge_reviewed_keeps(overrides, self.candidates, self.reviews)
        self.assertEqual(merged["keep"], ["1", "0"])
        merged = merge_reviewed_keeps(merged, self.candidates, self.reviews)
        self.assertEqual(merged["keep"], ["1", "0"])
        result = plan(self.candidates, self.probes, "60/1", overrides=merged)
        self.assertEqual([shot["id"] for shot in result["shots"]], ["0", "1"])

    def test_missing_empty_or_duplicate_review_key_is_rejected(self):
        for reviews in ([{"keep": True}], [{"key": "", "keep": True}], [{"key": "   ", "keep": True}],
                        [{"key": 3, "keep": True}], [{"key": "same"}, {"key": "same", "keep": True}]):
            with self.subTest(reviews=reviews), self.assertRaisesRegex(ValueError, "unique nonempty"):
                merge_reviewed_keeps({}, self.candidates, reviews)


if __name__ == "__main__":
    unittest.main()
