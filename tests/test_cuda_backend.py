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

from fpvsesh import cuda_backend, enhance, ai_validation
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
        self.ffmpeg = self.root / "ffmpeg.exe"
        self.ffprobe = self.root / "ffprobe.exe"
        self.ffmpeg.write_bytes(b"fake encoder")
        self.ffprobe.write_bytes(b"fake probe")
        self.stack.enter_context(patch.object(cuda_backend, "locate_tools", return_value=(str(self.ffmpeg), str(self.ffprobe))))
        self.relatives = ["fpvsesh/ai_models.py", "fpvsesh/ai_worker.py",
                     "fpvsesh/runtime_dlls.py",
                     "models/real-esrgan-cuda/RealESRGAN_x2plus.pth",
                     ".venv-ai/Lib/site-packages/torch/version.py"]
        for relative in self.relatives:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
        self.lock = self.root / "requirements-ai-lock.txt"
        self.lock.write_text("torch==2.14.0+cu126 --hash=sha256:" + "a" * 64 + "\n"
                             "numpy==2.2.6 --hash=sha256:" + "b" * 64 + "\n", encoding="utf-8")
        self.site = self.root / ".venv-ai/Lib/site-packages"
        self.torch_metadata = self.add_distribution("torch", "2.14.0+cu126")
        self.numpy_metadata = self.add_distribution("numpy", "2.2.6")

    def add_distribution(self, name, version, suffix=""):
        directory = self.site / f"{name}-{version}{suffix}.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
        (directory / "WHEEL").write_text("Wheel-Version: 1.0\nTag: cp312-cp312-win_amd64\n", encoding="utf-8")
        (directory / "RECORD").write_text(f"{name}/__init__.py,sha256=synthetic,1\n", encoding="utf-8")
        return directory

    def test_runtime_versions_are_read_from_only_the_isolated_environment(self):
        signature = cuda_backend.runtime_signature()
        self.assertEqual(set(signature["packages"]), {"numpy", "torch"})
        self.assertEqual(signature["packages"]["torch"]["version"], "2.14.0+cu126")
        # The main test interpreter also has NumPy, but that cannot satisfy a
        # missing package in the separate worker's site-packages directory.
        metadata_path = self.numpy_metadata / "METADATA"
        metadata_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "numpy must be installed"):
            cuda_backend.runtime_signature()

    def test_wrong_versions_missing_records_and_duplicate_distributions_reject(self):
        meta = self.torch_metadata / "METADATA"
        original = meta.read_text(encoding="utf-8")
        meta.write_text(original.replace("2.14.0+cu126", "2.8.0+cu126"), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "torch must be installed"):
            cuda_backend.runtime_signature()
        meta.write_text(original, encoding="utf-8")
        record = self.torch_metadata / "RECORD"
        record.unlink()
        with self.assertRaisesRegex(RuntimeError, "incomplete installed package metadata"):
            cuda_backend.runtime_signature()
        record.write_text("torch/module.py,sha256=test,1\n", encoding="utf-8")
        self.add_distribution("torch", "2.14.0+cu126", suffix="-duplicate")
        with self.assertRaisesRegex(RuntimeError, "torch must be installed exactly once"):
            cuda_backend.runtime_signature()

    def test_lock_and_installed_metadata_changes_retire_validation_receipts(self):
        gpu = {"name": "test GPU"}
        saved = {"passed": True, "signature": cuda_backend.signature(), "gpu": gpu}
        self.validation.write_text(json.dumps(saved), encoding="utf-8")
        self.assertTrue(cuda_backend.status(gpu)["available"])
        original_lock = self.lock.read_text(encoding="utf-8")
        self.lock.write_text(original_lock + "# Revised tested lock\n", encoding="utf-8")
        self.assertFalse(cuda_backend.status(gpu)["available"])
        self.lock.write_text(original_lock, encoding="utf-8")
        record = self.numpy_metadata / "RECORD"
        record.write_text(record.read_text(encoding="utf-8") + "numpy/changed.py,sha256=other,2\n", encoding="utf-8")
        self.assertFalse(cuda_backend.status(gpu)["available"])
        # A malformed saved JSON record must also report unavailable calmly.
        self.validation.write_text("[]", encoding="utf-8")
        self.assertFalse(cuda_backend.status(gpu)["available"])

    def test_validation_rejects_old_dependencies_before_source_or_gpu_work(self):
        source = self.root / "untouched.mp4"
        source.write_bytes(b"original source")
        metadata_path = self.torch_metadata / "METADATA"
        metadata_path.write_text("Name: torch\nVersion: 2.8.0+cu126\n", encoding="utf-8")
        before = list((self.root / "logs").iterdir())
        with (patch.object(ai_validation, "ROOT", self.root), patch.object(ai_validation, "probe") as probe,
              patch.object(ai_validation, "_detected_gpu") as gpu, patch.object(cuda_backend, "render") as render):
            with self.assertRaisesRegex(RuntimeError, "optional AI runtime"):
                ai_validation.validate_ai(source)
        probe.assert_not_called()
        gpu.assert_not_called()
        render.assert_not_called()
        self.assertEqual(list((self.root / "logs").iterdir()), before)
        self.assertEqual(source.read_bytes(), b"original source")

    def test_cpu_encoding_fallback_diagnostics_are_preserved_from_worker(self):
        source, output = self.root / "source.mp4", self.root / "restored.mp4"
        source.write_bytes(b"untouched original")
        warning = "GPU encoding unavailable; CPU encoding used while CUDA restoration remained enabled"
        process = SimpleNamespace(stdout=io.StringIO(""), returncode=0, poll=lambda: 0)
        def launch(command, **kwargs):
            output.write_bytes(b"completed mocked restoration")
            output.with_suffix(".mp4.ai.json").write_text(json.dumps(
                {"frames": 3, "encoder": "libx265", "encoder_preset": "fast",
                 "rate_control": "crf", "quality_value": 16, "warnings": [warning]}), encoding="utf-8")
            return process
        options = {"ffmpeg": str(self.ffmpeg), "ffprobe": str(self.ffprobe), "start": 0,
                   "duration": .05, "frames": 3, "fps": "60/1"}
        with patch.object(cuda_backend.subprocess, "Popen", side_effect=launch):
            result = cuda_backend.render(source, output, options, {"width": 2, "height": 2}, (4, 4), Mock())
        self.assertEqual(result["warnings"], [warning])
        self.assertEqual((result["encoder"], result["encoder_preset"], result["rate_control"]), ("libx265", "fast", "crf"))
        self.assertEqual(source.read_bytes(), b"untouched original")

    def test_validation_requires_matching_code_model_profile_and_hardware(self):
        gpu = {"name": "test GPU", "driver_version": "test driver", "uuid": "test identity"}
        self.assertFalse(cuda_backend.status(gpu)["available"])
        record = {"passed": True, "signature": cuda_backend.signature(), "gpu": gpu, "fps": 2.5}
        self.validation.write_text(json.dumps(record), encoding="utf-8")
        self.assertTrue(cuda_backend.status(gpu)["available"])
        self.assertEqual(cuda_backend.status(gpu)["fps"], 2.5)
        for relative in self.relatives:
            with self.subTest(changed_file=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                self.assertFalse(cuda_backend.status(gpu)["available"])
                self.assertIsNone(cuda_backend.status(gpu)["fps"])
                path.write_bytes(original)
        self.assertFalse(cuda_backend.status({**gpu, "driver_version": "different"})["available"])
        original_encoder = self.ffmpeg.read_bytes()
        self.ffmpeg.write_bytes(b"different encoder")
        self.assertFalse(cuda_backend.status(gpu)["available"])
        self.ffmpeg.write_bytes(original_encoder)
        with patch.object(cuda_backend, "PROFILE", {**cuda_backend.PROFILE, "ai_blend": .9}):
            self.assertFalse(cuda_backend.status(gpu)["available"])
        self.python.unlink()
        self.assertFalse(cuda_backend.status(gpu)["available"])

    def test_explicit_tool_overrides_require_their_own_matching_hashes(self):
        gpu = {"name": "test GPU", "driver_version": "test driver", "uuid": "test identity", "vram_mib": 8192}
        record = {"passed": True, "signature": cuda_backend.signature(), "gpu": gpu, "fps": 2.5}
        self.validation.write_text(json.dumps(record), encoding="utf-8")
        custom = self.root / "custom tools"
        custom.mkdir()
        overrides = {"ffmpeg": custom / "ffmpeg.exe", "ffprobe": custom / "ffprobe.exe"}
        for name, path in overrides.items():
            path.write_bytes(getattr(self, name).read_bytes())
        # A copy of the exact validated binaries is safe; their location is immaterial.
        self.assertTrue(cuda_backend.status(gpu, **overrides)["available"])
        for name, path in overrides.items():
            with self.subTest(tool=name):
                original = path.read_bytes()
                path.write_bytes(original + b" unvalidated replacement")
                self.assertTrue(cuda_backend.status(gpu)["available"])
                self.assertFalse(cuda_backend.status(gpu, **overrides)["available"])
                path.write_bytes(original)

        source, destination = self.root / "source.mp4", self.root / "output.mp4"
        source.write_bytes(b"original source bytes")
        destination.write_bytes(b"previous completed output")
        overrides["ffmpeg"].write_bytes(b"an encoder with different driver requirements")
        source_probe = {"width": 1440, "height": 1080, "avg_frame_rate": "60/1",
                        "duration": "10", "_format": {"duration": "10"}}
        with patch.object(enhance, "_probe", return_value=source_probe), \
                patch.object(enhance, "_detected_gpu", return_value=gpu), \
                patch.object(enhance, "BENCHMARK_SIGNATURE", self.root / "absent.json"), \
                patch.object(cuda_backend, "render") as render:
            with self.assertRaisesRegex(RuntimeError, "not validated"):
                enhance.enhance_segment(source, destination, {**overrides, "quality": "ai",
                    "start": 1, "duration": .05, "frames": 3, "fps": "60/1"}, Mock())
        render.assert_not_called()
        self.assertEqual(destination.read_bytes(), b"previous completed output")
        self.assertEqual(source.read_bytes(), b"original source bytes")
        self.assertFalse(list(self.root.glob("*.working-*")))

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
