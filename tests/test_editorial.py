"""Editorial regressions for terminal arrivals and confidence-weighted parallax.

These synthetic feature sequences test selection behavior. They do not prove
visual tree recognition, flight semantics, or reliable crash classification.
"""
from __future__ import annotations

import copy
import statistics
import unittest

from fpvsesh import analysis, planner


def feature_rows(duration=40, *, settled_start=None, pause=None, rotation=5,
                 proximity=.25, parallax_confidence=.9, residual_motion=1.5):
    rows = []
    for index in range(round(duration * 6)):
        time = index / 6
        stationary = ((settled_start is not None and time >= settled_start)
                      or (pause is not None and pause[0] <= time < pause[1]))
        # Two mild tail movements exceed the old .22 idle threshold but remain
        # consistent with a stable terminal image and the new motion limits.
        motion = (.3 if index % 11 == 0 else .08) if stationary else 6
        rows.append({
            "t": time, "motion": motion, "rotation": .01 if stationary else rotation,
            "dx": .03 if stationary else 2, "dy": .01 if stationary else 1,
            "sharpness": 600, "luma": .45, "contrast": .2, "black": .005, "white": .01,
            "hist": [.25] * 4,
            "hash": "0xa53cc35aa53cc35a" if stationary else hex((index * 0x1234571) & ((1 << 64) - 1)),
            "residual_motion": .03 if stationary else residual_motion,
            "proximity": .0 if stationary else proximity,
            "parallax_confidence": parallax_confidence,
            "foreground_texture": .5,
        })
    return rows


def analyzed(rows, name="synthetic-flight.mp4"):
    duration = rows[-1]["t"] + 1 / 6
    return {"source": name, "identity": name, "duration": duration, "rows": rows,
            "sample_fps": 6, "cuts_estimated": [], "version": "synthetic"}


class TerminalArrivalTests(unittest.TestCase):
    def test_terminal_settling_excludes_the_moving_arrival_before_the_stop(self):
        rows = feature_rows(settled_start=35)
        result = analysis.terminal_arrival(rows, 40)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["settled_start"], 35, delta=.5)
        self.assertAlmostEqual(result["exclude_start"], 25, delta=.5)
        self.assertGreater(result["confidence"], 0)
        self.assertTrue(result["evidence"])
        candidates = analysis.candidates_from_analysis([analyzed(rows)])
        arrivals = [c for c in candidates if c["end"] > 27 and c["end"] < 34]
        self.assertTrue(arrivals, "Fixture should contain active approach candidates before settling")
        self.assertTrue(all(c["arrival_estimate"] and c["unusable"] for c in arrivals))
        self.assertTrue(any(not c["unusable"] for c in candidates if c["end"] < 24))
        self.assertTrue(all(c["idle"] == 0 for c in arrivals))

    def test_rotation_without_terminal_settling_remains_valid_flight(self):
        rows = feature_rows(duration=24, rotation=35)
        # A completed maneuver has continued flight after the rotation burst;
        # a recording that ends while still spinning cannot prove recovery.
        for row in rows:
            if row["t"] >= 21:
                row["rotation"] = 2
        self.assertIsNone(analysis.terminal_arrival(rows, 24))
        candidates = analysis.candidates_from_analysis([analyzed(rows)], style="freestyle")
        self.assertTrue(candidates)
        self.assertTrue(all(not c["arrival_estimate"] for c in candidates))
        self.assertTrue(any(not c["unusable"] for c in candidates))

    def test_temporary_pause_with_subsequent_flight_is_not_a_terminal_arrival(self):
        rows = feature_rows(duration=30, pause=(17, 21))
        self.assertIsNone(analysis.terminal_arrival(rows, 30))
        candidates = analysis.candidates_from_analysis([analyzed(rows)])
        self.assertTrue(any(c["end"] > 25 and not c["unusable"] for c in candidates))

    def test_static_recording_is_not_classified_as_a_landing_without_active_flight(self):
        rows = feature_rows(duration=20, settled_start=0)
        self.assertIsNone(analysis.terminal_arrival(rows, 20))

    def test_explicit_keep_can_restore_an_estimated_arrival(self):
        data = analyzed(feature_rows(settled_start=35))
        candidates = analysis.candidates_from_analysis([data])
        arrival = next(c for c in candidates if c["arrival_estimate"] and 27 < c["end"] < 34)
        probe = {"source": data["source"], "fps": "60/1", "duration": data["duration"],
                 "frame_count": round(data["duration"] * 60), "time_base": "1/60000", "start_pts": 0}
        overrides = {"keep": [arrival["id"]], "exclude": [c["id"] for c in candidates if c is not arrival]}
        timeline = planner.plan(candidates, [probe], "60/1", "30", overrides=overrides)
        self.assertEqual([shot["id"] for shot in timeline["shots"]], [arrival["id"]])
        self.assertEqual(timeline["shots"][0]["selection_reason"], "user keep")


class ParallaxPreferenceTests(unittest.TestCase):
    def scores(self, *, proximity, confidence, residual):
        rows = feature_rows(duration=24, rotation=7, proximity=proximity,
                            parallax_confidence=confidence, residual_motion=residual)
        return [c["score"] for c in analysis.candidates_from_analysis([analyzed(rows)]) if not c["unusable"]]

    def test_local_parallax_signal_outscores_uniform_motion_when_quality_is_equal(self):
        global_motion = self.scores(proximity=.05, confidence=.9, residual=.1)
        local_parallax = self.scores(proximity=.9, confidence=.9, residual=5)
        self.assertTrue(global_motion and local_parallax)
        self.assertGreater(statistics.median(local_parallax), statistics.median(global_motion) + 15)

    def test_unreliable_parallax_signal_does_not_claim_a_proximity_bonus(self):
        low = self.scores(proximity=.05, confidence=0, residual=.1)
        high = self.scores(proximity=.9, confidence=0, residual=5)
        self.assertAlmostEqual(statistics.median(high), statistics.median(low), delta=1)


class ReviewedEditTests(unittest.TestCase):
    def test_reviewed_intervals_preserve_exact_bounds_and_reject_invalid_sources(self):
        data = analyzed(feature_rows(duration=30, rotation=12))
        reviewed = {"source": data["source"], "start": 3.25, "end": 18.75,
                    "key": "continuous-reviewed-line", "reason": "Keep both passes and their clean exit",
                    "confidence": .88}
        candidates = analysis.candidates_from_analysis([data], reviewed_intervals=[reviewed])
        chosen = [c for c in candidates if c.get("review_key") == reviewed["key"]]
        self.assertEqual(len(chosen), 1)
        self.assertEqual((chosen[0]["start"], chosen[0]["end"]), (3.25, 18.75))
        self.assertEqual(chosen[0]["end"] - chosen[0]["start"], 15.5)
        self.assertEqual(chosen[0]["confidence"], .88)
        self.assertIn("both passes", chosen[0]["reason"])
        for change in ({"source": "unknown-recording.mp4"}, {"start": -1},
                       {"end": 31}, {"start": 19, "end": 18.75}):
            invalid = {**reviewed, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                analysis.candidates_from_analysis([data], reviewed_intervals=[invalid])

    def test_editorial_order_requires_every_selected_id_exactly_once(self):
        data = analyzed(feature_rows(duration=30))
        reviewed = [{"source": data["source"], "start": start, "end": start + 4,
                     "key": str(index), "reason": "Reviewed continuous pass"}
                    for index, start in enumerate((1, 8, 15))]
        candidates = analysis.candidates_from_analysis([data], reviewed_intervals=reviewed)
        kept = [next(c["id"] for c in candidates if c.get("review_key") == str(index)) for index in range(3)]
        probe = {"source": data["source"], "fps": "60/1", "duration": data["duration"],
                 "frame_count": round(data["duration"] * 60), "time_base": "1/60000", "start_pts": 0}
        requested = [kept[2], kept[0], kept[1]]
        overrides = {"keep": kept, "exclude": [c["id"] for c in candidates if c["id"] not in kept],
                     "order": requested}
        timeline = planner.plan(copy.deepcopy(candidates), [probe], "60/1", "30", overrides=overrides)
        self.assertEqual([s["id"] for s in timeline["shots"]], requested)
        self.assertEqual(timeline["duration"], 12)
        for invalid in ([kept[0], kept[0], kept[2]], kept[:2],
                        [kept[0], kept[1], "unknown-id"], [*kept, "extra-id"], []):
            with self.subTest(order=invalid), self.assertRaisesRegex(ValueError, "exactly once"):
                planner.plan(copy.deepcopy(candidates), [probe], "60/1", "30",
                             overrides={**overrides, "order": invalid})


if __name__ == "__main__":
    unittest.main()
