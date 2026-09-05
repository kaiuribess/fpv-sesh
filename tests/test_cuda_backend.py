"""CUDA validation and subprocess cancellation contracts, with no GPU required."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fpvsesh import cuda_backend, enhance
from fpvsesh.control import Cancelled


class CudaBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name in ("cache", "logs"):
            (self.root / name).mkdir()
        self.python = self.root / "python.exe"
        self.python.write_bytes(b"isolated test environment placeholder")
        self.validation = self.root / "logs/validation.json"
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(cuda_backend, "ROOT", self.root))
        self.stack.enter_context(patch.object(cuda_backend, "PYTHON", self.python))
        self.stack.enter_context(patch.object(cuda_backend, "VALIDATION", self.validation))

    def test_validation_requires_matching_code_model_profile_and_hardware(self):
        relatives = ["fpvsesh/ai_models.py", "fpvsesh/ai_worker.py",
                     "models/real-esrgan-cuda/RealESRGAN_x2plus.pth",
                     ".venv-ai/Lib/site-packages/torch/version.py"]
        for relative in relatives:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
        gpu = {"name": "test GPU", "driver_version": "test driver", "uuid": "test identity"}
        self.assertFalse(cuda_backend.status(gpu)["available"])
        record = {"passed": True, "signature": cuda_backend.signature(), "gpu": gpu, "fps": 2.5}
        self.validation.write_text(json.dumps(record), encoding="utf-8")
        self.assertTrue(cuda_backend.status(gpu)["available"])
        self.assertEqual(cuda_backend.status(gpu)["fps"], 2.5)
        for relative in relatives:
            with self.subTest(changed_file=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                self.assertFalse(cuda_backend.status(gpu)["available"])
                self.assertIsNone(cuda_backend.status(gpu)["fps"])
                path.write_bytes(original)
        self.assertFalse(cuda_backend.status({**gpu, "driver_version": "different"})["available"])
        with patch.object(cuda_backend, "PROFILE", {**cuda_backend.PROFILE, "ai_blend": .9}):
            self.assertFalse(cuda_backend.status(gpu)["available"])
        self.python.unlink()
        self.assertFalse(cuda_backend.status(gpu)["available"])

    def test_exit75_propagates_through_enhancement_and_removes_only_partial_output(self):
        source, destination = self.root / "source.mp4", self.root / "output.mp4"
        source.write_bytes(b"original source bytes")
        destination.write_bytes(b"previous completed output")
        stream = io.StringIO(json.dumps({"stage": "cancelled", "message": "stopped"}) + "\n")
        process = SimpleNamespace(stdout=stream, returncode=75, poll=lambda: 75)
        launched = []
        def launch(command, **kwargs):
            config = json.loads(Path(command[-1]).read_text())
            launched.append(config)
            Path(config["output"]).write_bytes(b"unfinished segment")
            return process
        source_probe = {"width": 1440, "height": 1080, "avg_frame_rate": "60/1",
                        "duration": "10", "_format": {"duration": "10"}}
        options = {"width": 64, "height": 36, "frames": 3, "fps": "60/1",
                   "start": 1, "duration": .05, "quality": "ai", "codec": "hevc"}
        with patch.object(enhance, "_probe", return_value=source_probe), \
                patch.object(enhance, "backend_status", return_value={"ai_available": True}), \
                patch.object(cuda_backend.subprocess, "Popen", side_effect=launch):
            with self.assertRaises(Cancelled):
                enhance.enhance_segment(source, destination, options, Mock())
        self.assertEqual(len(launched), 1)
        self.assertEqual(launched[0]["source"], str(source))
        self.assertFalse(Path(launched[0]["output"]).exists())
        self.assertFalse(list((self.root / "cache").glob("cuda-*.json")))
        self.assertFalse(destination.with_suffix(".mp4.enhance.json").exists())
        self.assertEqual(destination.read_bytes(), b"previous completed output")
        self.assertEqual(source.read_bytes(), b"original source bytes")
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
