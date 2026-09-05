import copy
from types import SimpleNamespace
import unittest

from fpvsesh.pacing import favor_beats
from fpvsesh.planner import plan
from fpvsesh.settings import resolve_settings


def fixture():
    probe = {"source": "flight.mp4", "fps": "60/1", "duration": 30, "frame_count": 1800,
             "time_base": "1/60", "start_pts": 0}
    candidates = [{"id": str(index), "source": "flight.mp4", "identity": "same-flight",
                   "start": start, "end": end, "score": 90 - index, "unusable": False,
                   "hist": [.5, .5], "hash_sequence": [hex(index * 9876551)] * 3,
                   "source_duration": 30, "rotation": 0, "dx_in": 1, "dx_out": 1}
                  for index, (start, end) in enumerate(((0, 5), (10, 15)))]
    timeline = plan(candidates, [probe], "60/1", "auto", overrides={"order": ["0", "1"]})
    rows = [{"t": i/6, "rotation": 0, "dy": 0, "motion": 3} for i in range(180)]
    return timeline, [probe], [{"source": "flight.mp4", "rows": rows}], {"beats": [5.5, 11], "confidence": .9}


class MusicalPacingTests(unittest.TestCase):
    def test_safe_exit_extended_to_beat_without_changing_approach_or_later_source(self):
        timeline, probes, analyses, music = fixture()
        result = favor_beats(timeline, probes, analyses, music)
        self.assertEqual([(s["start"], s["end"]) for s in result["shots"]], [(0, 5.5), (10, 15)])
        self.assertEqual(result["frames"], 630)
        self.assertEqual(result["shots"][1]["timeline_in_frame"], 330)
        self.assertEqual(timeline["frames"], 600)

    def test_linked_trick_in_extension_keeps_original_cut_instead_of_cutting_new_burst(self):
        timeline, probes, analyses, music = fixture()
        analyses[0]["rows"][32]["rotation"] = 35
        result = favor_beats(timeline, probes, analyses, music)
        self.assertEqual(result["frames"], 600)

    def test_reviewed_and_user_kept_passages_remain_exact(self):
        for protection in ({"review_key": "exact"}, {"selection_reason": "user keep"}):
            timeline, probes, analyses, music = fixture()
            timeline["shots"][0].update(protection)
            self.assertEqual(favor_beats(timeline, probes, analyses, music)["frames"], 600)

    def test_duration_limit_arrival_overlap_and_unclear_music_block_extension(self):
        timeline, probes, analyses, music = fixture()
        timeline["target"] = "10"
        self.assertEqual(favor_beats(timeline, probes, analyses, music)["frames"], 600)
        timeline["target"] = "auto"
        analyses[0]["terminal_arrival"] = {"exclude_start": 5.2}
        self.assertEqual(favor_beats(timeline, probes, analyses, music)["frames"], 600)
        analyses[0].pop("terminal_arrival")
        music["confidence"] = .1
        self.assertEqual(favor_beats(timeline, probes, analyses, music)["frames"], 600)
        music["confidence"] = .9
        timeline["shots"][1]["start"] = 5.2
        self.assertEqual(favor_beats(timeline, probes, analyses, music)["frames"], 600)

    def test_chronological_order_preserves_source_order_and_exact_intervals(self):
        timeline, probes, _, _ = fixture()
        candidates = copy.deepcopy(timeline["shots"])
        candidates[0]["score"] = 45
        candidates[1]["score"] = 95
        result = plan(candidates, probes, "60/1", edit_order="chronological")
        self.assertEqual([shot["start"] for shot in result["shots"]], [0, 10])


class SettingsTests(unittest.TestCase):
    def test_music_removal_resume_and_social_formats_are_explicit(self):
        saved = {"music": "old.mp3", "social_formats": ["vertical"], "music_offset": 3.0}
        settings = resolve_settings(SimpleNamespace(no_music=True), saved)
        self.assertIsNone(settings["music"])
        self.assertEqual(settings["social_formats"], ["vertical"])
        self.assertEqual(settings["music_offset"], 3)
        settings = resolve_settings(SimpleNamespace(social_formats="vertical,square,vertical"), {})
        self.assertEqual(settings["social_formats"], ["vertical", "square"])
        self.assertEqual(resolve_settings(SimpleNamespace(social_formats="none"), saved)["social_formats"], [])

    def test_nonfinite_values_and_invalid_saved_choices_fail_before_render(self):
        for key in ("music_level", "music_offset", "music_fade", "recovery", "focus_x"):
            for invalid in (float("nan"), float("inf"), -1, "oops"):
                with self.assertRaises(ValueError):
                    resolve_settings(SimpleNamespace(), {key: invalid})
        with self.assertRaises(ValueError):
            resolve_settings(SimpleNamespace(), {"framing": "invented"})


if __name__ == "__main__":
    unittest.main()
