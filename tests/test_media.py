"""Small real-media tests; these prove timing/IO behavior, not FPV aesthetics."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fpvsesh import media


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg, cls.ffprobe = media.locate_tools()
        cls.temp = tempfile.TemporaryDirectory(prefix="fpvsesh-media-")
        cls.folder = Path(cls.temp.name)
        cls.silent = cls.fixture("flight café's [A] & $HOME.mp4", "60/1", 36)
        cls.ntsc = cls.fixture("ntsc.mp4", "60000/1001", 36)
        cls.slower = cls.fixture("30fps.mp4", "30/1", 18)
        cls.audio = cls.fixture("with audio.mp4", "60/1", 36, audio=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def fixture(cls, name, fps, frames, audio=False, black=False):
        output = cls.folder / name
        src = f"color=c=black:s=160x90:r={fps}" if black else f"testsrc2=size=160x90:rate={fps}"
        command = [cls.ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i", src]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "0.6"]
        command += ["-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-x264-params",
                    "colorprim=bt709:transfer=bt709:colormatrix=bt709"]
        command += ["-c:a", "aac"] if audio else ["-an"]
        media.run(command + [str(output)])
        return output

    def test_probe_handles_unicode_apostrophes_and_shell_characters(self):
        before = self.silent.read_bytes()
        result = media.probe(self.silent)
        self.assertEqual((result["width"], result["height"]), (160, 90))
        self.assertEqual(result["fps"], "60/1")
        self.assertEqual(result["frame_count"], 36)
        self.assertFalse(result["audio"])
        self.assertFalse(result["hdr"])
        self.assertEqual(result["color_transfer"], "bt709")
        self.assertEqual(result["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(self.silent.read_bytes(), before)

    def test_audio_presence_and_alignment(self):
        self.assertTrue(media.probe(self.audio)["audio"])
        result = media.validate_output(self.audio, 36, "60/1", 160, 90)
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["audio_alignment"][0]["aligned_within_120ms"])

    def test_fractional_rate_is_preserved(self):
        result = media.probe(self.ntsc)
        self.assertEqual(result["fps"], "60000/1001")
        self.assertEqual(media.choose_fps([result, result]), "60000/1001")
        verified = media.validate_output(self.ntsc, 36, "60000/1001", 160, 90)
        self.assertTrue(verified["passed"], verified["errors"])

    def test_mixed_rates_have_documented_duration_preserving_conversion(self):
        result = media.fps_decision([media.probe(self.ntsc), media.probe(self.slower)])
        self.assertEqual(result["fps"], "60/1")
        self.assertEqual(len(result["conversions"]), 2)
        self.assertIn("no interpolation", result["conversions"][1]["method"])
        with self.assertRaises(ValueError):
            media.choose_fps([])

    def test_full_frame_count_and_monotonic_pts(self):
        result = media.inspect_timestamps(self.silent, include_pts=True)
        self.assertEqual(result["frame_count"], 36)
        self.assertTrue(result["strictly_monotonic"])
        self.assertTrue(result["scan_complete"])
        self.assertFalse(result["vfr"])
        self.assertEqual(result["gap_count"], 0)
        self.assertEqual(len(result["pts"]), 36)
        self.assertFalse(result["corruption_detected"])
        self.assertAlmostEqual(result["last_time"], 35 / 60)

    def test_validator_rejects_wrong_timeline_expectations(self):
        result = media.validate_output(self.silent, 40, "30/1", 320, 180)
        self.assertFalse(result["passed"])
        self.assertGreaterEqual(len(result["errors"]), 4)

    def test_black_intervals_are_review_flags_without_false_automatic_failure(self):
        fixture = self.fixture("dark shot.mp4", "60/1", 36, black=True)
        result = media.validate_output(fixture, 36, "60/1", 160, 90)
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["black_intervals"])
        self.assertTrue(result["warnings"])
        self.assertFalse(result["visual_quality_reviewed"])

    def test_invalid_and_missing_media_fail_clearly(self):
        with self.assertRaises(FileNotFoundError):
            media.probe(self.folder / "missing.mp4")
        empty = self.folder / "empty.mp4"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty"):
            media.probe(empty)
        invalid = self.folder / "invalid.mp4"
        invalid.write_bytes(b"this is not video")
        with self.assertRaises(RuntimeError):
            media.probe(invalid)
        self.assertEqual(invalid.read_bytes(), b"this is not video")

    def test_timestamp_scan_detects_missing_repeated_and_reversed_pts(self):
        metadata = media.probe(self.silent, include_hash=False)
        output = "pts=0|best_effort_timestamp=0\npts=256\npts=256\npts=128\npts=N/A|best_effort_timestamp=N/A\npts=1024\n"
        with patch.object(media, "probe", return_value=metadata), patch.object(
                media, "run", return_value=subprocess.CompletedProcess([], 0, output, "")):
            result = media.inspect_timestamps(self.silent)
        self.assertFalse(result["strictly_monotonic"])
        self.assertEqual(result["duplicate_pts_count"], 1)
        self.assertEqual(result["non_monotonic_count"], 1)
        self.assertEqual(result["missing_pts_count"], 1)
        self.assertTrue(result["vfr"])

    def test_variable_frame_intervals_are_detected(self):
        output = self.folder / "vfr.mp4"
        media.run([self.ffmpeg, "-v", "error", "-nostdin", "-i", str(self.silent),
                   "-vf", "setpts=PTS+if(gte(N\\,18)\\,0.05/TB\\,0)",
                   "-fps_mode", "vfr", "-an", "-c:v", "libx264", str(output)])
        result = media.inspect_timestamps(output)
        self.assertTrue(result["vfr"])
        self.assertGreaterEqual(result["gap_count"], 1)

    def test_run_rejects_shell_string_and_logs_failure(self):
        with self.assertRaises(TypeError):
            media.run("echo unwanted")
        logfile = self.folder / "failure log.txt"
        result = media.run([self.ffprobe, "-not_an_option"], logfile, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Exit code:", logfile.read_text(encoding="utf-8"))

    def test_hdr_metadata_is_explicitly_flagged(self):
        output = self.folder / "pq metadata.mp4"
        media.run([self.ffmpeg, "-v", "error", "-nostdin", "-i", str(self.silent),
                   "-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
                   "-c:v", "libx264", "-x264-params",
                   "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
                   "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", str(output)])
        result = media.probe(output)
        self.assertTrue(result["hdr"])
        self.assertEqual(result["color_transfer"], "smpte2084")


if __name__ == "__main__":
    unittest.main()
