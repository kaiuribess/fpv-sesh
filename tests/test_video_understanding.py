"""Protect ordinary flight, abstention, source-bound caches and exact cuts."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fpvsesh import video_understanding as video
from fpvsesh.video_worker import decode_window
from fpvsesh.rotation_witness import _decode as decode_rotation
from fpvsesh.media import locate_tools, run, probe


def answer(label="ordinary flight", **flags):
    value = {"label": label, "evidence": "The camera moves forward through the scene.",
             "complete": True, "recovery": True, "inversion": False,
             "obstacle_relation": False, "impact": False}
    value.update(flags)
    return json.dumps(value)


class ObservationTests(unittest.TestCase):
    def test_ordinary_flight_is_valid_without_inventing_tricks(self):
        item = video.interpret(answer(), 0, 8)
        self.assertEqual((item["label"], item["status"]), ("ordinary flight", "suggested"))
        self.assertNotIn("confidence", item)
        # Omitted model flags never invent a full rotation or recovery.
        missing = video.interpret('{"label":"ordinary flight","evidence":"Steady forward flight."}', 0, 8)
        self.assertEqual(missing["label"], "ordinary flight")
        self.assertFalse(missing["observations"]["inversion"])

    def test_acrobatics_need_all_visible_completion_checks(self):
        for label in video.ACROBATICS:
            valid = answer(label, inversion=True, obstacle_relation=True)
            self.assertEqual(video.interpret(valid, 0, 8)["label"], label)
            for key in ("complete", "recovery", "inversion"):
                with self.subTest(label=label, missing=key):
                    data = json.loads(valid)
                    data[key] = False
                    item = video.interpret(json.dumps(data), 0, 8)
                    self.assertEqual(item["label"], "uncertain")
                    self.assertEqual(item["raw_label"], label)

    def test_tree_and_obstacle_names_require_relationship_evidence(self):
        for label in ("powerloop", "tree weaving", "orbit"):
            self.assertEqual(video.interpret(answer(label, inversion=True), 0, 8)["label"], "uncertain")
        self.assertEqual(video.interpret(answer("split-S", inversion=True), 0, 8)["label"], "split-S")

    def test_stationary_tail_is_not_automatically_a_crash(self):
        self.assertEqual(video.interpret(answer("crash"), 0, 8)["label"], "uncertain")
        self.assertEqual(video.interpret(answer("crash", impact=True), 0, 8)["label"], "crash")

    def test_malformed_output_remains_inert_uncertain_data(self):
        for raw in ('run command.exe', '[]', '{"label":"double-flip"}', answer("roll", inversion="true"), 'x'*7000):
            self.assertEqual(video.interpret(raw, 0, 8)["label"], "uncertain")

    def test_candidate_context_does_not_certify_a_partial_cut(self):
        observation = video.interpret(answer("roll", inversion=True), 0, 8)
        data = {"video_events": [observation]}
        self.assertEqual(video.candidate_observation({"start": 0, "end": 8}, data)["status"], "suggested")
        self.assertEqual(video.candidate_observation({"start": 1, "end": 7}, data)["status"], "uncertain")
        self.assertEqual(video.candidate_observation({"start": .2, "end": 7.8}, data)["status"], "uncertain")
        self.assertIsNone(video.candidate_observation({"start": 20, "end": 28}, data))

    def test_disagreeing_acrobatics_remain_uncertain(self):
        events = [video.interpret(answer(label, inversion=True), start, start+8)
                  for label, start in (("roll", 0), ("flip", 7))]
        item = video.candidate_observation({"start": 0, "end": 15}, {"video_events": events})
        self.assertEqual(item["status"], "uncertain")

    def test_measured_roll_is_a_suggestion_with_distinct_provenance(self):
        observed = video.interpret(answer(), 0, 8)
        measured = {"full_source_window": True, "valid_fraction": .95,
                    "bursts": [{"start": 2, "end": 4, "signed_degrees": 359,
                                "complete_image_rotation": True, "after_seconds": 3}]}
        combined = video.combine_evidence(observed, measured)
        self.assertEqual((combined["label"], combined["status"]), ("roll", "suggested"))
        self.assertEqual(combined["raw_label"], "ordinary flight")
        self.assertIn("not independently", combined["checks"][-1])
        self.assertIn("does not prove airborne", combined["evidence"])
        self.assertEqual(observed["label"], "ordinary flight")
        measured["bursts"][0]["after_seconds"] = .2
        self.assertEqual(video.combine_evidence(observed, measured)["status"], "uncertain")
        measured["valid_fraction"] = .4
        self.assertEqual(video.combine_evidence(observed, measured)["label"], "ordinary flight")

    def test_off_clears_previous_model_results_without_loading_model(self):
        analyses = [{"video_events": [{"label": "roll"}]}]
        with patch.object(video, "model_status", side_effect=AssertionError("model must stay unloaded")):
            result = video.recognize(analyses, "unused", mode="off")
        self.assertFalse(result["available"])
        self.assertNotIn("video_events", analyses[0])

    def test_malformed_optional_model_manifest_is_unavailable(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(video, "ROOT", Path(folder)):
            path = Path(folder) / "models/qwen3-vl-2b/manifest.json"
            for data in ([], {"assets": None}, {"assets": [None]}, {"assets": ["bad"]}):
                video._save(path, data)
                self.assertFalse(video.model_status()["available"])

    def test_cache_cannot_relabel_an_observation_or_transfer_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observation.json"
            signature = {"source": "original-sha", "window": {"start": 0, "end": 1}, "profile": {"sample_fps": 2, "max_frames": 16}}
            raw = answer()
            record = {"signature": signature, "event": video.interpret(raw, 0, 1), "raw_response": raw,
                      "sampled_frames": 2, "sample_times": [0, .5], "inference_demonstrated": True}
            video._save(path, record)
            self.assertIsNotNone(video._cached(path, signature))
            changed = {**signature, "source": "replacement-sha"}
            self.assertIsNone(video._cached(path, changed))
            tampered = copy.deepcopy(record)
            tampered["event"]["label"] = "roll"
            video._save(path, tampered)
            self.assertIsNone(video._cached(path, signature))
            shortened = copy.deepcopy(record)
            shortened["sampled_frames"] = 1
            shortened["sample_times"] = [0]
            video._save(path, shortened)
            self.assertIsNone(video._cached(path, signature))
            record["sample_times"] = [0, 0]
            video._save(path, record)
            self.assertIsNone(video._cached(path, signature))


class SourceFrameTests(unittest.TestCase):
    def test_fractional_source_tails_cover_exact_grid_at_8_16_and_30_fps(self):
        with tempfile.TemporaryDirectory(prefix="fractional-video-tail-") as folder:
            ffmpeg, _ = locate_tools()
            for index, (rate, count, start) in enumerate((("60", 299, 0), ("60000/1001", 79, .25), ("60", 61, .125))):
                path = Path(folder) / f"fractional-{index}.mp4"
                run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=red:s=160x120:r={rate}",
                     "-frames:v", str(count), "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
                duration = probe(path, include_hash=False)["duration"]
                for fps in (8, 16):
                    with self.subTest(source=index, fps=fps):
                        frames, times = decode_window(path, {"start": start, "end": duration},
                                                     {"sample_fps": fps, "max_frames": 96}, ffmpeg)
                        expected = math.ceil((duration-start)*fps-1e-6)
                        self.assertEqual(len(frames), expected)
                        self.assertLess(times[-1], duration-start)
                frames = decode_rotation(path, start, duration, ffmpeg, lambda: None)
                self.assertEqual(len(frames), math.ceil((duration-start)*30-1e-6))
                # The frame cap must not pad genuinely absent source material.
                with self.assertRaisesRegex(ValueError, "ended early"):
                    decode_rotation(path, start, duration+.2, ffmpeg, lambda: None)

    def test_source_trim_excludes_later_content_before_resampling(self):
        with tempfile.TemporaryDirectory(prefix="bounded-video-window-") as folder:
            ffmpeg, _ = locate_tools()
            path = Path(folder) / "later-blue.mp4"
            run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=160x120:r=60:d=1",
                 "-f", "lavfi", "-i", "color=blue:s=160x120:r=60:d=1",
                 "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
            for fps in (8, 16):
                frames, _ = decode_window(path, {"start": .125, "end": .991},
                                          {"sample_fps": fps, "max_frames": 96}, ffmpeg)
                self.assertTrue((frames[:, 128, 128, 0] > 200).all())
                self.assertTrue((frames[:, 128, 128, 2] < 10).all())

    def test_decode_preserves_aspect_and_relative_video_timestamps(self):
        with tempfile.TemporaryDirectory(prefix="video-理解-") as folder:
            path = Path(folder) / "portrait.mp4"
            ffmpeg, _ = locate_tools()
            run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                 "color=red:s=120x240:r=30:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
            before = path.read_bytes()
            frames, times = decode_window(path, {"start": .5, "end": 1.5},
                                          {"sample_fps": 8, "max_frames": 8}, ffmpeg)
            self.assertEqual(frames.shape, (8, 256, 256, 3))
            self.assertEqual(times, [i/8 for i in range(8)])
            self.assertLess(frames[:, :, :50].max(), 5)
            self.assertGreater(frames[:, 128, 128, 0].min(), 200)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "ended early"):
                decode_window(path, {"start": 1, "end": 3}, {"sample_fps": 8, "max_frames": 16}, ffmpeg)


if __name__ == "__main__":
    unittest.main()
