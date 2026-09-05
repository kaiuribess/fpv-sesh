"""Behavioral tests for timeline selection, safe interruption and real previews."""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fpvsesh import media, planner, render


def source(name="source.mp4", fps="60/1", duration=60):
    return {"source": name, "sha256": name, "fps": fps, "duration": duration,
            "frame_count": round(duration * float(Fraction(fps))),
            "time_base": "1/60000", "start_time": 0}


def candidate(p, name, start, end, score=80, signature="0xffffffffffffffff"):
    return {"id": name, "source": p["source"], "identity": p["sha256"],
            "start": start, "end": end, "score": score, "unusable": False,
            "hash_sequence": [signature] * 3, "hist": [0.5, 0.5],
            "source_duration": p["duration"], "rotation": .4,
            "dx_in": 1, "dx_out": 1, "luma": .36}


class PlannerTests(unittest.TestCase):
    def test_overlap_and_identical_file_copies_are_not_repeated(self):
        p = source()
        entries = [candidate(p, "a", 1, 8, 90), candidate(p, "b", 4, 11, 89, "0x0"),
                   candidate(p, "c", 20, 27, 75, "0xaaaaaaaaaaaaaaaa")]
        q = source("renamed duplicate.mp4")
        q["sha256"] = p["sha256"]
        entries.append(candidate(q, "d", 1, 8, 88))
        timeline = planner.plan(entries, [p, q], "60/1", "30")
        self.assertEqual({s["id"] for s in timeline["shots"]}, {"a", "c"})
        self.assertEqual(timeline["frames"], 840)
        self.assertEqual(timeline["duration"], 14)

    def test_explicit_keep_exclude_and_conflict_checks(self):
        p = source()
        entries = [candidate(p, "a", 1, 8, 90), candidate(p, "b", 9, 16, 20, "0x0")]
        timeline = planner.plan(copy.deepcopy(entries), [p], "60/1", "30", overrides={"keep": ["b"], "exclude": ["a"]})
        self.assertEqual([s["id"] for s in timeline["shots"]], ["b"])
        self.assertEqual(timeline["shots"][0]["selection_reason"], "user keep")
        for overrides in ({"keep": ["a"], "exclude": ["a"]}, {"keep": ["missing"]}):
            with self.assertRaises(ValueError):
                planner.plan(copy.deepcopy(entries), [p], "60/1", "30", overrides=overrides)
        entries[1]["start"] = 2
        with self.assertRaises(ValueError):
            planner.plan(entries, [p], "60/1", "30", overrides={"keep": ["a", "b"]})

    def test_short_session_is_not_padded_and_intervals_are_contiguous(self):
        p = source(fps="60000/1001")
        entries = [candidate(p, "a", .3, 5.3), candidate(p, "b", 8.2, 15.2, signature="0x0")]
        timeline = planner.plan(entries, [p], "60000/1001", "90")
        self.assertLess(timeline["duration"], 13)
        self.assertEqual(len(timeline["shots"]), 2)
        cursor = 0
        for shot in timeline["shots"]:
            self.assertEqual(shot["timeline_in_frame"], cursor)
            self.assertGreater(shot["frames"], 0)
            self.assertGreaterEqual(shot["start"], 0)
            self.assertLessEqual(shot["end"], p["duration"])
            cursor = shot["timeline_out_frame"]
            self.assertEqual(Fraction(shot["source_start_time"]) * Fraction(p["fps"]), shot["source_start_frame"])
        self.assertEqual(cursor, timeline["frames"])
        self.assertAlmostEqual(timeline["duration"], cursor * 1001 / 60000)

    def test_weak_footage_does_not_fill_the_target(self):
        p = source()
        weak = candidate(p, "weak", 1, 8, 10)
        with self.assertRaisesRegex(ValueError, "No usable"):
            planner.plan([weak], [p], "60/1", "60")
        kept = planner.plan([weak], [p], "60/1", "60", overrides={"keep": ["weak"]})
        self.assertEqual(kept["duration"], 7)

    def test_rate_conversion_preserves_elapsed_source_time(self):
        p = source(fps="30/1")
        timeline = planner.plan([candidate(p, "slow", 1, 8)], [p], "60/1", "30")
        self.assertEqual(timeline["duration"], 7)
        self.assertEqual(timeline["frames"], 420)
        self.assertTrue(timeline["shots"][0]["rate_conversion"])
        self.assertEqual(timeline["shots"][0]["source_end_frame_exclusive"] - timeline["shots"][0]["source_start_frame"], 210)

    def test_low_disk_space_is_rejected_before_render(self):
        with patch.object(render.shutil, "disk_usage", return_value=SimpleNamespace(free=100)):
            with self.assertRaisesRegex(RuntimeError, "Not enough free disk"):
                render.ensure_space(Path.cwd(), 200)
            render.ensure_space(Path.cwd(), 100)

    def test_vfr_uses_actual_pts_and_can_include_the_final_frame(self):
        p = source(fps="25/1", duration=.2)
        p.update({"time_base": "1/1000", "start_time": .5, "start_pts": 500,
                  "frame_pts": [500, 520, 570, 620, 660]})
        timeline = planner.plan([candidate(p, "variable", .025, .2)], [p], "60/1", "30")
        shot = timeline["shots"][0]
        self.assertEqual(shot["source_start_frame"], 2)
        self.assertEqual(shot["source_end_frame_exclusive"], 5)
        self.assertEqual(shot["source_pts_start"], "570")
        self.assertEqual(shot["source_pts_end_exclusive"], "700")
        self.assertEqual(Fraction(shot["source_start_time"]), Fraction(7, 100))
        self.assertEqual(Fraction(shot["source_end_time"]), Fraction(1, 5))
        self.assertEqual(shot["frames"], 8)
        self.assertTrue(shot["rate_conversion"])

    def test_out_of_bounds_or_empty_intervals_are_rejected(self):
        p = source()
        for start, end in ((61, 62), (2, 2), (-1, 2), (59, 61), (1, float("nan"))):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "source bounds"):
                planner.plan([candidate(p, "invalid", start, end)], [p], "60/1", "30")
        with self.assertRaisesRegex(ValueError, "no complete timeline frames"):
            planner.plan([candidate(p, "tiny", .001, .002)], [p], "60/1", "30")

    def test_target_stops_selection_without_repeating_or_trimming_moments(self):
        p = source()
        signatures = ["0xffffffffffffffff", "0x0", "0xaaaaaaaaaaaaaaaa", "0x5555555555555555", "0x00000000ffffffff"]
        entries = [candidate(p, str(i), i * 10, i * 10 + 7, 90 - i, sig) for i, sig in enumerate(signatures)]
        timeline = planner.plan(entries, [p], "60/1", "30")
        self.assertEqual(timeline["duration"], 28)
        self.assertEqual(len({shot["id"] for shot in timeline["shots"]}), 4)


class PreviewRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="fpvsesh-pipeline-")
        cls.folder = Path(cls.temp.name)
        ffmpeg, _ = media.locate_tools()
        cls.paths = []
        for index, rate in enumerate(("60/1", "60000/1001")):
            path = cls.folder / f"flight café's [{index}] & dollar.mp4"
            command = [ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate={rate}"]
            if index == 0:
                command += ["-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000", "-t", "0.6"]
            command += ["-frames:v", "36", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709"]
            command += ["-c:a", "aac"] if index == 0 else ["-an"]
            media.run(command + [str(path)])
            cls.paths.append(path)
        cls.probes = [media.probe(p) for p in cls.paths]
        cls.original_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths]
        cls.settings = {"look": "natural", "strength": 0, "quality": "lanczos", "codec": "h264"}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def prepare(self, label):
        job = self.folder / label
        cache = job / "cache"
        cache.mkdir(parents=True)
        entries = [candidate(self.probes[0], "a", 0, .5),
                   candidate(self.probes[1], "b", 0, .5, signature="0x0")]
        timeline = planner.plan(entries, self.probes, "60/1", "30")
        return job, cache, timeline

    def test_real_multiclip_preview_contains_edited_audio_and_preserves_originals(self):
        job, cache, timeline = self.prepare("audio preview")
        render.make_audio(timeline, self.probes, job, .4, lambda *a: None, lambda: None)
        result = render.render_timeline(timeline, self.probes, self.settings, job, cache,
                                        lambda *a: None, lambda: None, preview=True)
        self.assertTrue(result["verification"]["passed"])
        self.assertTrue(result["verification"]["probe"]["audio"])
        self.assertEqual(result["verification"]["timestamps"]["frame_count"], 60)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths], self.original_hashes)

    def test_cancel_after_completed_segment_resumes_without_encoding_that_segment(self):
        job, cache, timeline = self.prepare("interrupted preview")
        calls = 0
        def cancel_at_second_shot():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InterruptedError("test cancellation at supported boundary")
        with self.assertRaises(InterruptedError):
            render.render_timeline(timeline, self.probes, self.settings, job, cache,
                                   lambda *a: None, cancel_at_second_shot, preview=True)
        completed = list((cache / "segments").glob("*.mp4"))
        self.assertEqual(len(completed), 1)
        before = completed[0].stat().st_mtime_ns
        checkpoint = json.loads((job / "preview-checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["completed_frames"], 30)
        result = render.render_timeline(timeline, self.probes, self.settings, job, cache,
                                        lambda *a: None, lambda: None, preview=True)
        self.assertTrue(result["verification"]["passed"])
        self.assertEqual(completed[0].stat().st_mtime_ns, before)

    def test_orphan_segment_without_checkpoint_record_is_rebuilt(self):
        job, cache, timeline = self.prepare("orphan preview")
        render.render_timeline(timeline, self.probes, self.settings, job, cache,
                               lambda *a: None, lambda: None, preview=True)
        segment = next((cache / "segments").glob("*.mp4"))
        segment.with_suffix(".json").unlink()
        segment.write_bytes(b"interrupted encode")
        result = render.render_timeline(timeline, self.probes, self.settings, job, cache,
                                        lambda *a: None, lambda: None, preview=True)
        self.assertTrue(result["verification"]["passed"])
        self.assertTrue(segment.with_suffix(".json").exists())
        self.assertEqual(media.probe(segment)["frame_count"], 30)

    def test_corrupt_packet_cache_with_intact_metadata_is_rebuilt(self):
        job, cache, timeline = self.prepare("corrupt packet preview")
        render.render_timeline(timeline, self.probes, self.settings, job, cache,
                               lambda *a: None, lambda: None, preview=True)
        segment = next((cache / "segments").glob("*.mp4"))
        content = bytearray(segment.read_bytes())
        marker = content.index(b"mdat")
        box_size = int.from_bytes(content[marker - 4:marker], "big")
        self.assertGreater(box_size, 8)
        content[marker + 4:marker - 4 + box_size] = b"\0" * (box_size - 8)
        segment.write_bytes(content)
        # Intact MP4 metadata makes shallow checkpoint validation pass.
        self.assertEqual(media.probe(segment)["frame_count"], 30)
        result = render.render_timeline(timeline, self.probes, self.settings, job, cache,
                                        lambda *a: None, lambda: None, preview=True)
        self.assertTrue(result["verification"]["passed"])
        self.assertEqual(media.inspect_timestamps(segment)["frame_count"], 30)


if __name__ == "__main__":
    unittest.main()
