"""End-to-end CLI tests in an isolated application copy with real FFmpeg media."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

from fpvsesh import media


class CliEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="fpvsesh-cli-")
        cls.folder = Path(cls.temp.name)
        cls.app = cls.folder / "application"
        cls.app.mkdir()
        shutil.copytree(media.PROJECT_ROOT / "fpvsesh", cls.app / "fpvsesh",
                        ignore=shutil.ignore_patterns("__pycache__"))
        for directory in ("input", "music", "output", "cache", "models", "logs"):
            (cls.app / directory).mkdir()
        cls.ffmpeg, cls.ffprobe = media.locate_tools()
        cls.paths = []
        # Both clips have full-image camera-like movement. A static test chart
        # would reasonably be rejected by flight/idle heuristics.
        for index, (rate, frames) in enumerate((("60/1", 312), ("60000/1001", 324))):
            path = cls.folder / f"飞行🚁 café's [{index}] & $HOME.mp4"
            # Finish with a continued horizontal flight line long enough for
            # the maneuver recovery guard; don't end mid-turn or on a freeze.
            movement = ("crop=320:240:x=160+140*sin(min(t\\,2)*2.7)+max(t-2\\,0)*25:"
                        "y=120+100*cos(min(t\\,2)*3.1)")
            if index:
                movement += ",hflip"
            # A fixed, textured scene makes the camera path the sole motion:
            # testsrc2's independently animated shapes would otherwise keep
            # introducing new bursts after the intended recovery starts.
            scene = (f"testsrc2=size=640x480:rate={rate},noise=alls=22:all_seed=17,"
                     f"trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/(({rate})*TB)")
            command = [cls.ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i", scene]
            if index == 0:
                command += ["-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000", "-t", "5.2"]
            command += ["-vf", movement, "-frames:v", str(frames), "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-x264-params",
                        "colorprim=bt709:transfer=bt709:colormatrix=bt709"]
            command += ["-c:a", "aac"] if index == 0 else ["-an"]
            media.run(command + [str(path)])
            cls.paths.append(path)
        cls.hashes = [media.sha256_file(path) for path in cls.paths]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_cli(self, args, cancel_at_proxy=False):
        env = os.environ.copy()
        env["FPVSESH_FFMPEG"] = self.ffmpeg
        env["FPVSESH_FFPROBE"] = self.ffprobe
        # Exercise the documented direct CLI, without the UI's UTF-8 override.
        # main() must safely configure its own output even with a legacy locale.
        env["PYTHONIOENCODING"] = "cp1252"
        env.pop("PYTHONPATH", None)
        process = subprocess.Popen(
            [sys.executable, "-m", "fpvsesh.cli", *args], cwd=self.app, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="strict",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        events, output, reader_errors = [], [], []
        sent_cancel = False
        def read_events():
            nonlocal sent_cancel
            try:
                for line in process.stdout:
                    output.append(line)
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    events.append(event)
                    if cancel_at_proxy and not sent_cancel and event.get("stage") == "proxy":
                        control = self.app / "cache" / "control.json"
                        temporary = control.with_suffix(".tmp")
                        temporary.write_text('{"action":"cancel"}', encoding="utf-8")
                        temporary.replace(control)
                        sent_cancel = True
            except Exception as exc:
                reader_errors.append(exc)
        reader = threading.Thread(target=read_events, daemon=True)
        reader.start()
        try:
            process.wait(timeout=120)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            reader.join(timeout=10)
            process.stdout.close()
        self.assertFalse(reader_errors, reader_errors)
        self.assertFalse(reader.is_alive(), "CLI event reader did not finish")
        self.assertEqual(process.returncode, 0, "".join(output)[-7000:])
        return events, "".join(output), sent_cancel

    def test_short_unicode_multiclip_full_pipeline_cancel_and_resume(self):
        job = self.app / "output" / "short-session"
        args = ["make", "--job", str(job), "--duration", "30", "--quality", "lanczos", "--preview-only"]
        for path in self.paths:
            args += ["--input", str(path)]
        first, _, cancelled = self.run_cli(args, cancel_at_proxy=True)
        self.assertTrue(cancelled, "Test never reached the real proxy stage")
        self.assertTrue(any(event.get("stage") == "cancelled" for event in first))
        self.assertEqual(json.loads((job / "status.json").read_text(encoding="utf-8"))["stage"], "cancelled")
        self.assertFalse((job / "preview.mp4").exists())
        self.assertFalse((self.app / "cache" / "control.json").exists())

        # Resume from the real saved settings and source identity/timestamp cache.
        events, output, _ = self.run_cli(["make", "--job", str(job), "--preview-only"])
        self.assertEqual(events[-1]["stage"], "complete", output[-5000:])
        self.assertTrue(any("飞行🚁" in event.get("message", "") for event in events))
        self.assertNotIn("UnicodeEncodeError", output)
        self.assertNotIn("�", output)
        timeline = json.loads((job / "timeline.json").read_text(encoding="utf-8"))
        sources = json.loads((job / "sources.json").read_text(encoding="utf-8"))
        verification = json.loads((job / "preview-verification.json").read_text(encoding="utf-8"))
        analyses = json.loads((job / "analysis-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(sources), 2)
        self.assertEqual(len(analyses), 2)
        candidate_data = json.loads((job / "candidates.json").read_text(encoding="utf-8"))["candidates"]
        selection_details = [{key: item.get(key) for key in
                              ("source", "score", "start", "end", "selected", "unusable", "idle",
                               "recovery_incomplete", "reason", "selection_reason", "hash_sequence")}
                             for item in candidate_data]
        self.assertEqual(len({shot["source"] for shot in timeline["shots"]}), 2, selection_details)
        self.assertGreaterEqual(len(timeline["shots"]), 2)
        self.assertLess(timeline["duration"], 11)
        self.assertTrue(timeline["fps_decision"]["conversions"])
        self.assertTrue(verification["passed"], verification["errors"])
        self.assertEqual(verification["timestamps"]["frame_count"], timeline["frames"])
        self.assertEqual(verification["probe"]["fps"], "60/1")
        self.assertEqual((verification["probe"]["width"], verification["probe"]["height"]), (1280, 720))
        self.assertTrue(verification["probe"]["audio"])
        self.assertIsNone(timeline["music"])
        self.assertTrue((job / "report.md").is_file())
        self.assertEqual([media.sha256_file(path) for path in self.paths], self.hashes)
        for analysis in analyses:
            self.assertGreaterEqual(analysis["coverage_seconds"], analysis["duration"] - .1)
        saved_settings = json.loads((job / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_settings["inputs"], [str(path.resolve()) for path in self.paths])


if __name__ == "__main__":
    unittest.main()
