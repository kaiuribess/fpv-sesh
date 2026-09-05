"""Protect linked motion bursts and complete kept intervals without padding.

Feature thresholds are conservative editing cues, not proof of a specific
maneuver, crash, or completed upright recovery.
"""
from __future__ import annotations

import copy
import unittest

from fpvsesh import analysis, planner
from test_editorial import analyzed, feature_rows


def burst_rows(duration=20, bursts=()):
    rows = feature_rows(duration=duration, rotation=2)
    for time, rotation, vertical, motion in bursts:
        row = rows[round(time * 6)]
        row.update({"rotation": rotation, "dy": vertical, "motion": motion})
    return rows


class RecoveryEndTests(unittest.TestCase):
    def test_completed_recovery_and_unrelated_later_burst_leave_end_unchanged(self):
        rows = burst_rows(bursts=[(3, 25, 0, 6), (11, 20, 0, 6)])
        self.assertEqual(analysis.guard_recovery_end(rows, 0, 9, 20), 9)

    def test_rotation_is_protected_even_when_optical_flow_is_low(self):
        rows = burst_rows(bursts=[(8, -148, 0, 4)])
        self.assertEqual(analysis.guard_recovery_end(rows, 1, 9, 20), 10.5)

    def test_quiet_samples_between_linked_bursts_do_not_end_a_double_maneuver(self):
        rows = burst_rows(bursts=[(8, 12, 0, 6), (10, -9, 0, 4),
                                 (12, 2, 11, 12), (17, 40, 0, 6)])
        # The second burst lies beyond the original cut but within its hold;
        # the third lies inside the extended hold and restarts it again.
        self.assertEqual(analysis.guard_recovery_end(rows, 1, 9, 20), 14.5)

    def test_vertical_burst_requires_motion_support_and_hold_is_configurable(self):
        supported = burst_rows(bursts=[(8, 1, -11, 8)])
        weak = burst_rows(bursts=[(8, 1, -11, 7.9)])
        self.assertEqual(analysis.guard_recovery_end(supported, 1, 9, 20), 10.5)
        self.assertEqual(analysis.guard_recovery_end(weak, 1, 9, 20), 9)
        self.assertEqual(analysis.guard_recovery_end(supported, 1, 9, 20, hold=3), 11)
        with self.assertRaises(ValueError):
            analysis.guard_recovery_end(supported, 1, 9, 20, hold=0)

    def test_insufficient_source_or_arrival_limit_returns_none_without_shortening_hold(self):
        rows = burst_rows(duration=12, bursts=[(9.5, 15, 0, 6)])
        self.assertEqual(analysis.guard_recovery_end(rows, 1, 10, 12), 12)
        self.assertIsNone(analysis.guard_recovery_end(rows, 1, 10, 11.9))
        self.assertIsNone(analysis.guard_recovery_end(rows, 1, 10, 10.5))

    def test_incomplete_maneuver_candidates_remain_reviewable_but_not_automatic(self):
        data = analyzed(feature_rows(duration=12, rotation=35))
        candidates = analysis.candidates_from_analysis([data], style="freestyle")
        self.assertTrue(candidates)
        self.assertTrue(all(c["recovery_incomplete"] and c["unusable"] for c in candidates))
        self.assertTrue(all(not c["arrival_estimate"] for c in candidates))
        probe = {"source": data["source"], "fps": "60/1", "duration": 12,
                 "frame_count": 720, "time_base": "1/60000", "start_pts": 0}
        chosen = candidates[0]
        overrides = {"keep": [chosen["id"]],
                     "exclude": [c["id"] for c in candidates if c is not chosen]}
        timeline = planner.plan(candidates, [probe], "60/1", "30", overrides=overrides)
        self.assertEqual([shot["id"] for shot in timeline["shots"]], [chosen["id"]])

    def test_reviewed_exact_bounds_bypass_automatic_recovery_extension(self):
        data = analyzed(feature_rows(duration=12, rotation=35))
        reviewed = {"source": data["source"], "start": 3.25, "end": 9.75,
                    "key": "reviewed-boundaries", "reason": "Specific reviewed source interval"}
        candidates = analysis.candidates_from_analysis([data], reviewed_intervals=[reviewed])
        result = next(c for c in candidates if c.get("review_key") == reviewed["key"])
        self.assertEqual((result["start"], result["end"]), (3.25, 9.75))
        self.assertFalse(result["recovery_incomplete"])
        self.assertIsNone(result["recovery_hold_seconds"])

    def test_reviewed_bounds_cannot_transfer_to_replaced_recording(self):
        data = analyzed(feature_rows(duration=12, rotation=35))
        data["identity"] = "original-content-hash"
        reviewed = {"source": data["source"], "source_identity": data["identity"],
                    "start": 3.25, "end": 9.75, "key": "confirmed-double-flip"}
        candidates = analysis.candidates_from_analysis([data], reviewed_intervals=[reviewed])
        confirmed = next(c for c in candidates if c.get("review_key") == reviewed["key"])
        self.assertEqual((confirmed["start"], confirmed["end"]), (3.25, 9.75))
        data["identity"] = "replacement-content-hash"
        with self.assertRaisesRegex(ValueError, "recording has changed"):
            analysis.candidates_from_analysis([data], reviewed_intervals=[reviewed])

    def test_short_exact_ranges_survive_automatic_minimum_and_deduplication(self):
        data = analyzed(feature_rows(duration=12, rotation=35))
        ranges = [{"source": data["source"], "start": start, "end": end, "key": key}
                  for start, end, key in [(1.25, 3.5, "short"), (1.3, 3.6, "nearby"),
                                          (5.02, 5.05, "between-analysis-samples")]]
        candidates = analysis.candidates_from_analysis([data], reviewed_intervals=ranges)
        reviewed = {c["review_key"]: c for c in candidates if c.get("review_key")}
        self.assertEqual(set(reviewed), {r["key"] for r in ranges})
        for expected in ranges:
            candidate = reviewed[expected["key"]]
            self.assertEqual((candidate["start"], candidate["end"]), (expected["start"], expected["end"]))
            self.assertTrue(candidate["hash_sequence"])


class KeptDurationTests(unittest.TestCase):
    def test_auto_preserves_kept_intervals_over_75_seconds_without_filler_or_trimming(self):
        source = "long-source.mp4"
        probe = {"source": source, "fps": "60/1", "duration": 120,
                 "frame_count": 7200, "time_base": "1/60000", "start_pts": 0}
        candidates = []
        for index, (start, end) in enumerate(((0, 32), (34, 66), (68, 100), (108, 112))):
            candidates.append({"id": str(index), "source": source, "identity": source,
                               "start": start, "end": end, "score": 90 - index,
                               "unusable": False, "hash_sequence": [hex(0x1234 << index)] * 3,
                               "hist": [.5, .5], "source_duration": 120, "rotation": .2,
                               "end_rotation": 2, "dx_in": 1, "dx_out": 1})
        overrides = {"keep": ["0", "1", "2"], "order": ["0", "1", "2"]}
        timeline = planner.plan(copy.deepcopy(candidates), [probe], "60/1", "auto", overrides=overrides)
        self.assertEqual(timeline["duration"], 96)
        self.assertEqual(timeline["frames"], 5760)
        self.assertEqual([s["id"] for s in timeline["shots"]], ["0", "1", "2"])
        self.assertEqual([(s["start"], s["end"]) for s in timeline["shots"]], [(0, 32), (34, 66), (68, 100)])
        with self.assertRaisesRegex(ValueError, "exceed duration target"):
            planner.plan(copy.deepcopy(candidates), [probe], "60/1", "90", overrides=overrides)


if __name__ == "__main__":
    unittest.main()
