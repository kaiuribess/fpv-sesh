"""Coverage, abstention, and truthful temporal support for video proposals."""
import math
import unittest

from fpvsesh.maneuvers import (
    TAXONOMY_SOURCES, TRICK_DEFINITIONS, TRICK_LABELS, build_windows, motion_evidence,
)


def rows(duration=4, rotation=0, motion=4):
    return [{"t": index / 6, "rotation": rotation, "motion": motion,
             "proximity": .7, "parallax_confidence": .8}
            for index in range(int(duration * 6))]


class ManeuverWindowTests(unittest.TestCase):
    def test_whole_flight_coverage_and_fractional_tail(self):
        for mode, width, overlap in (("auto", 8, 1), ("thorough", 6, 2)):
            for duration in (.01, 1, 6, 8, 8.001, 15, 39.983333, 295.233333):
                windows = build_windows({"duration": duration}, mode)
                self.assertEqual(windows[0]["start"], 0)
                self.assertEqual(windows[-1]["end"], duration)
                self.assertTrue(all(0 <= w["start"] < w["end"] <= duration for w in windows))
                self.assertTrue(all(w["end"] - w["start"] <= width for w in windows))
                for left, right in zip(windows, windows[1:]):
                    self.assertEqual(left["end"] - right["start"], overlap)

    def test_estimated_cuts_do_not_bisect_fast_tricks(self):
        self.assertEqual(build_windows({"duration": 16, "cuts_estimated": [2, 6, 8]}),
                         [{"start": 0.0, "end": 8.0}, {"start": 7.0, "end": 15.0},
                          {"start": 14.0, "end": 16.0}])
        self.assertEqual(build_windows({"duration": 0}), [])
        self.assertEqual(build_windows({"duration": 8}, "off"), [])

    def test_invalid_window_settings(self):
        for duration in (True, -1, math.inf, math.nan, "8", None):
            with self.assertRaises(ValueError):
                build_windows({"duration": duration})
        with self.assertRaises(ValueError):
            build_windows({"duration": 8}, "fastest")


class MotionEvidenceTests(unittest.TestCase):
    def test_ordinary_flight_does_not_get_a_named_trick_or_probability(self):
        result = motion_evidence(rows(rotation=2), 0, 4)
        self.assertEqual(result["status"], "measured")
        self.assertFalse(result["rotation_present"])
        self.assertEqual(result["motion_level"], "moving")
        self.assertEqual(result["stationary_fraction"], 0)
        self.assertEqual(result["tracking_quality"], "unverified")
        self.assertNotIn("label", result)
        self.assertNotIn("confidence", result)
        self.assertNotIn("probability", result)
        self.assertNotIn("flip_count", result)

    def test_signed_image_rotation_preserves_cancellation_without_counting_rolls(self):
        data = rows(rotation=20)
        for row in data[12:]:
            row["rotation"] = -20
        result = motion_evidence(data, 0, 4)
        self.assertTrue(result["rotation_present"])
        self.assertEqual(result["rotation_sign_changes"], 1)
        self.assertEqual(result["signed_image_rotation_degrees"], -20)
        self.assertEqual(result["absolute_image_rotation_degrees"], 460)
        self.assertEqual(result["sampling_fps_estimate"], 6)
        self.assertTrue(any("cannot establish" in text for text in result["limitations"]))

    def test_unmeasured_tracking_failures_never_become_stationary_evidence(self):
        data = rows(rotation=100, motion=0)
        for row in data:
            row.update(rotation_valid=False, motion_valid=False)
        result = motion_evidence(data, 0, 4)
        self.assertEqual(result["status"], "unmeasured")
        self.assertIsNone(result["rotation_present"])
        self.assertIsNone(result["stationary_fraction"])
        self.assertEqual(result["motion_level"], "unknown")
        self.assertEqual(result["tracking_quality"], "reported")

    def test_missing_and_nonfinite_fields_are_not_silently_zero(self):
        data = [{"t": 0}, {"t": 1 / 6, "motion": math.nan, "rotation": math.inf},
                {"t": 2 / 6, "motion": True, "rotation": False}, None, {"t": math.nan}]
        result = motion_evidence(data, 0, 1)
        self.assertIsNone(result["stationary_fraction"])
        self.assertIsNone(result["signed_image_rotation_degrees"])
        self.assertIsNone(result["close_pass_present"])
        self.assertEqual(result["status"], "unmeasured")

    def test_interval_boundaries_duplicates_and_long_gaps_break_accumulation(self):
        data = [{"t": 0, "motion": 1, "rotation": 90},
                {"t": .1, "motion": 1, "rotation": 20},
                {"t": .2, "motion": 1, "rotation": 30},
                {"t": .2, "motion": 1, "rotation": 30},
                {"t": .3, "motion": 1, "rotation": 40},
                {"t": 2, "motion": 1, "rotation": 50}]
        result = motion_evidence(data, .05, 3)
        self.assertIsNone(result["signed_image_rotation_degrees"])
        self.assertGreater(result["maximum_sample_gap_seconds"], .5)
        self.assertEqual(result["status"], "unmeasured")

    def test_short_interval_without_complete_pair_abstains(self):
        for data in ([], [{"t": .1, "motion": 0, "rotation": 0}], rows()):
            result = motion_evidence(data, .05, .15)
            self.assertEqual(result["status"], "unmeasured")
            self.assertIsNone(result["rotation_sign_changes"])

    def test_unknown_rotation_breaks_sign_change_evidence(self):
        data = rows(rotation=20)
        data[12]["rotation_valid"] = False
        for row in data[13:]:
            row["rotation"] = -20
        result = motion_evidence(data, 0, 4)
        self.assertEqual(result["rotation_sign_changes"], 0)

    def test_thresholds_tolerate_frame_timestamp_rounding(self):
        result = motion_evidence(rows(rotation=8, motion=.4), 0, 4)
        self.assertTrue(result["rotation_present"])
        self.assertEqual(result["stationary_fraction"], 0)
        self.assertEqual(result["motion_level"], "moving")

    def test_sampling_rate_normalization_preserves_support(self):
        slow = rows(rotation=10, motion=3)
        fast = [{"t": index / 30, "rotation": 2, "motion": .6}
                for index in range(120)]
        low_rate = motion_evidence(slow, 0, 4)
        high_rate = motion_evidence(fast, 0, 4)
        self.assertEqual(low_rate["rotation_present"], high_rate["rotation_present"])
        self.assertEqual(low_rate["motion_level"], high_rate["motion_level"])
        self.assertEqual(low_rate["median_motion_pixels_per_second"], high_rate["median_motion_pixels_per_second"])

    def test_low_motion_is_not_a_landing_or_crash_classification(self):
        result = motion_evidence(rows(motion=.1), 0, 4)
        self.assertEqual(result["stationary_fraction"], 1)
        self.assertEqual(result["motion_level"], "low")
        self.assertNotIn("label", result)
        self.assertTrue(any("hovering" in text for text in result["limitations"]))

    def test_bounds_and_grounded_vocabulary(self):
        for start, end in ((True, 1), (-1, 1), (1, 1), (1, math.inf), (0, math.nan)):
            with self.assertRaises(ValueError):
                motion_evidence([], start, end)
        self.assertEqual(set(TRICK_LABELS), set(TRICK_DEFINITIONS))
        self.assertIn("ordinary flight", TRICK_LABELS)
        self.assertIn("uncertain", TRICK_LABELS)
        self.assertTrue(all(source["url"].startswith("https://") for source in TAXONOMY_SOURCES))


if __name__ == "__main__":
    unittest.main()
