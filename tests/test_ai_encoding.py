"""Encoder preflight/fallback contracts; real CPU encoding, no model or GPU work."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2  # Initialize the extension before the temporary sys.modules model stub.
import numpy as np

from fpvsesh.control import Cancelled
from fpvsesh.media import locate_tools


class AiEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models = ModuleType("fpvsesh.ai_models")
        models.Restorer = Mock()
        path = Path(__file__).resolve().parents[1] / "fpvsesh/ai_worker.py"
        spec = importlib.util.spec_from_file_location("fpvsesh._tested_encoding_worker", path)
        cls.worker = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"fpvsesh.ai_models": models}):
            spec.loader.exec_module(cls.worker)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="encoding-飞行-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "original ' 飞行.mkv"
        self.source.write_bytes(b"untouched source")
        self.config = {"source": str(self.source), "output": str(self.root / "result.mp4"),
                       "ffmpeg": "test-ffmpeg", "frames": 3, "fps": "60000/1001", "start": 0,
                       "source_width": 320, "source_height": 180,
                       "content_width": 640, "content_height": 360, "width": 640, "height": 360,
                       "ai_model": "RealESRGAN_x2plus", "ai_blend": 1, "codec": "hevc", "cq": 16}
        self.log = self.root / "encoder-probe.log"
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(self.worker, "ROOT", self.root))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_successful_nvenc_probe_keeps_actual_quality_and_does_not_try_cpu(self):
        with patch.object(self.worker.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr=b"")) as run:
            result = self.worker._select_encoder("ffmpeg with spaces", self.config, Mock(), self.log)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[0], "ffmpeg with spaces")
        self.assertEqual(command[command.index("-c:v") + 1], "hevc_nvenc")
        self.assertEqual(command[command.index("-frames:v") + 1], "3")
        self.assertEqual(command[command.index("-framerate") + 1], "60000/1001")
        self.assertEqual(len(run.call_args.kwargs["input"]), 320 * 180 * 3 * 3)
        self.assertLessEqual(run.call_args.kwargs["timeout"], 20)
        self.assertEqual(result["encoder_preset"], "p7")
        self.assertEqual(result["rate_control"], "cq")
        self.assertEqual(result["cq"], 16)
        self.assertIsNone(result["crf"])
        self.assertFalse(result["encoder_fallback"])
        self.assertEqual(result["warnings"], [])

    def test_failed_nvenc_uses_codec_appropriate_cpu_options_and_truthful_metadata(self):
        for codec, encoder, pixel in (("hevc", "libx265", "yuv420p10le"), ("h264", "libx264", "yuv420p")):
            with self.subTest(codec=codec), patch.object(self.worker.subprocess, "run", side_effect=[
                    SimpleNamespace(returncode=1, stderr=b"Driver does not support NVENC API"),
                    SimpleNamespace(returncode=0, stderr=b"")]) as run:
                result = self.worker._select_encoder("ffmpeg", {**self.config, "codec": codec}, Mock(), self.log)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-c:v") + 1], encoder)
            self.assertIn("format=" + pixel, command[command.index("-vf") + 1])
            for option in ("-cq", "-rc", "-tune", "-multipass", "-temporal-aq", "-tier"):
                self.assertNotIn(option, command)
            if codec == "hevc":
                self.assertEqual(command[command.index("-profile:v") + 1], "main10")
            self.assertEqual(result["encoder_preset"], "medium")
            self.assertEqual(result["rate_control"], "crf")
            self.assertIsNone(result["cq"])
            self.assertEqual(result["crf"], 16)
            self.assertTrue(result["encoder_fallback"])
            self.assertIn("CUDA restoration remains enabled", result["warnings"][0])
            self.assertIn("Driver does not support", self.log.read_text())

    def test_both_encoder_checks_fail_before_model_or_source_decode(self):
        with patch.object(self.worker.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr=b"unsupported")) as run, \
                patch.object(self.worker, "Restorer") as model, patch.object(self.worker.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "GPU and CPU video encoder checks both failed"):
                self.worker.render(self.config)
        self.assertEqual(run.call_count, 2)
        model.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(self.source.read_bytes(), b"untouched source")
        self.assertFalse(Path(self.config["output"]).exists())

    def test_cancel_after_probe_is_not_treated_as_encoder_failure(self):
        checkpoint = Mock(side_effect=[None, Cancelled("stopped")])
        with patch.object(self.worker.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr=b"driver")) as run:
            with self.assertRaises(Cancelled):
                self.worker._select_encoder("ffmpeg", self.config, checkpoint, self.log)
        run.assert_called_once()

    def test_unavailable_executable_is_fatal_without_repeated_attempts(self):
        with patch.object(self.worker.subprocess, "run", side_effect=FileNotFoundError("missing")) as run:
            with self.assertRaisesRegex(RuntimeError, "Could not start"):
                self.worker._select_encoder("missing", self.config, Mock(), self.log)
        run.assert_called_once()

    def test_probe_timeout_is_bounded_and_can_fall_back(self):
        with patch.object(self.worker.subprocess, "run", side_effect=[subprocess.TimeoutExpired("ffmpeg", 20),
                SimpleNamespace(returncode=0, stderr=b"")]):
            result = self.worker._select_encoder("ffmpeg", self.config, Mock(), self.log)
        self.assertEqual(result["encoder"], "libx265")
        self.assertIn("timed out after 20 seconds", self.log.read_text())

    def test_cuda_model_failure_is_never_retried_as_software_restoration(self):
        plan = self.worker._encoder_plan(self.config, software=True)
        with patch.object(self.worker, "_select_encoder", return_value=plan) as select, \
                patch.object(self.worker, "Restorer", side_effect=RuntimeError("CUDA inference unavailable")) as model, \
                patch.object(self.worker.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "CUDA inference unavailable"):
                self.worker.render(self.config)
        select.assert_called_once()
        model.assert_called_once()
        popen.assert_not_called()
        self.assertFalse(Path(self.config["output"]).with_suffix(".mp4.ai.json").exists())

    def test_real_cpu_worker_encode_preserves_format_frame_count_color_and_source(self):
        try:
            ffmpeg, ffprobe = locate_tools()
        except (OSError, RuntimeError, ValueError) as error:
            self.skipTest(f"Verified bundled tools are not installed: {error}")
        run_real = subprocess.run
        options = {"capture_output": True, "timeout": 30,
                   "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        created = run_real([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
                           "testsrc2=size=320x180:rate=60000/1001", "-frames:v", "3", "-c:v", "ffv1",
                           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                           "-color_range", "tv", str(self.source)], **options)
        self.assertEqual(created.returncode, 0, created.stderr.decode(errors="replace"))
        source_sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        calls = []
        def probe_without_gpu(command, **kwargs):
            calls.append(command)
            if "hevc_nvenc" in command or "h264_nvenc" in command:
                return SimpleNamespace(returncode=1, stderr=b"Synthetic unavailable GPU encoder")
            return run_real(command, **kwargs)
        restored = []
        def enhance(original, outscale):
            restored.append(original.copy())
            return np.repeat(np.repeat(original, 2, axis=0), 2, axis=1)
        restorer = SimpleNamespace(enhance=enhance, close=Mock())
        for codec, expected, pixel in (("hevc", "hevc", "yuv420p10le"), ("h264", "h264", "yuv420p")):
            with self.subTest(codec=codec):
                output = self.root / f"CPU ' 飞行-{codec}.mp4"
                with patch.object(self.worker.subprocess, "run", side_effect=probe_without_gpu), \
                        patch.object(self.worker, "Restorer", return_value=restorer):
                    result = self.worker.render({**self.config, "ffmpeg": ffmpeg, "output": str(output), "codec": codec})
                meta = run_real([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(output)], **options)
                self.assertEqual(meta.returncode, 0, meta.stderr.decode(errors="replace"))
                video = json.loads(meta.stdout)["streams"][0]
                self.assertEqual((video["width"], video["height"]), (640, 360))
                self.assertEqual(video["codec_name"], expected)
                self.assertEqual(video["pix_fmt"], pixel)
                self.assertEqual(video["avg_frame_rate"], "60000/1001")
                self.assertEqual(int(video["nb_read_frames"]), 3)
                self.assertEqual(video["color_range"], "tv")
                for key in ("color_space", "color_transfer", "color_primaries"):
                    self.assertEqual(video[key], "bt709")
                if codec == "hevc":
                    self.assertEqual(video["profile"], "Main 10")
                decoded = run_real([ffmpeg, "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"], **options)
                self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors="replace"))
                self.assertEqual(result["encoder"], "libx265" if codec == "hevc" else "libx264")
                self.assertTrue(result["ai_inference"])
                self.assertTrue(result["encoder_fallback"])
                self.assertEqual(result["frames"], 3)
                self.assertEqual(json.loads(output.with_suffix(".mp4.ai.json").read_text())["encoder"], result["encoder"])
        self.assertEqual(len(restored), 6)
        self.assertEqual(restorer.close.call_count, 2)
        self.assertEqual(len(calls), 4)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), source_sha)


if __name__ == "__main__":
    unittest.main()
