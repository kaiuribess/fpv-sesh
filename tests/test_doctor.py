"""Readiness checks must remain lightweight, honest, and safe to share."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fpvsesh import doctor


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="private-personal-workspace-")
        self.root = Path(self.directory.name)
        (self.root / ".venv/Scripts").mkdir(parents=True)
        (self.root / ".venv/Scripts/python.exe").touch()
        (self.root / "requirements-lock.txt").write_text("numpy==2.2.6 --hash=sha256:example\npillow==12.3.0\n")
        tools = self.root / "tools/current"
        tools.mkdir(parents=True)
        for name in ("ffmpeg", "ffprobe", "ffplay"):
            (tools / (name + ".exe")).write_bytes(b"placeholder never executed")
        self.manifest = {"tools": [{"name": "FFmpeg", "required": True, "version": "9.0.1-full_build",
            "executable": "tools/current/ffmpeg.exe", "probe_executable": "tools/current/ffprobe.exe",
            "player_executable": "tools/current/ffplay.exe"}]}
        self.save(self.root / "tools/dependencies.json", self.manifest)
        self.runtime = {"python": "3.13.15", "bits": 64, "supported": True, "tk_available": True,
                        "tk_version": "8.6", "windows": True}
        self.patchers = [patch.object(doctor, "_runtime", side_effect=lambda: dict(self.runtime)),
                         patch.object(doctor.sys, "prefix", str(self.root / ".venv")),
                         patch.object(doctor.metadata, "version", side_effect=lambda name: {"numpy": "2.2.6", "pillow": "12.3.0"}[name]),
                         patch.object(doctor, "bundled_tool", side_effect=lambda kind, root: Path(root) / "tools/current" / (kind + ".exe")),
                         patch.object(doctor, "_execute", side_effect=self.execute)]
        self.mocks = [patcher.start() for patcher in self.patchers]

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.directory.cleanup()

    @staticmethod
    def save(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def execute(command):
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "NVIDIA GeForce RTX 4060 Ti, 581.42, 8192\n", "")
        name = Path(command[0]).stem
        return subprocess.CompletedProcess(command, 0, f"{name} version 9.0.1-full_build\n", "")

    def test_required_ready_optional_absent_still_supports_basic_editing(self):
        report = doctor.collect_report(self.root)
        self.assertTrue(report["ready"])
        self.assertTrue(all(not model["files_present"] for model in report["optional_models"]))
        self.assertEqual(report["gpu"]["encode_status"], "not benchmarked by doctor")
        self.assertIn("CPU fallback", report["gpu"]["message"])
        self.assertEqual(len(self.mocks[-1].call_args_list), 3)
        self.assertTrue(all(call.args[0][0] == "nvidia-smi" or call.args[0][1:] == ["-version"]
                            for call in self.mocks[-1].call_args_list))

    def test_missing_required_package_tool_tk_or_environment_fails_readiness(self):
        for component in ("packages", "tools", "tk", "environment"):
            with self.subTest(component=component):
                patches = {"packages": patch.object(doctor.metadata, "version", return_value="0.0"),
                           "tools": patch.object(doctor, "_execute", side_effect=OSError("private path should not escape")),
                           "tk": patch.object(doctor, "_runtime", return_value={**self.runtime, "tk_available": False}),
                           "environment": patch.object(doctor.sys, "prefix", str(self.root / "other-runtime"))}
                with patches[component]:
                    report = doctor.collect_report(self.root)
                self.assertFalse(report["ready"])
                self.assertFalse(next(item for item in report["checks"] if item["id"] == component)["passed"])

    def test_old_tool_version_is_not_accepted_as_current(self):
        def old(command):
            return subprocess.CompletedProcess(command, 0, f"{Path(command[0]).stem} version 7.1.1\n", "")
        with patch.object(doctor, "_execute", side_effect=old):
            report = doctor.collect_report(self.root)
        self.assertFalse(report["ready"])
        self.assertEqual(report["tools"][0]["version"], "7.1.1")
        self.assertFalse(report["tools"][0]["matches_manifest_version"])

    def test_manifest_escape_is_not_executed(self):
        self.manifest["tools"][0]["executable"] = "../outside.exe"
        self.save(self.root / "tools/dependencies.json", self.manifest)
        report = doctor.collect_report(self.root)
        self.assertFalse(report["ready"])
        self.assertTrue(all("outside.exe" not in call.args[0][0] for call in self.mocks[-1].call_args_list))

    def test_tool_integrity_failure_prevents_executing_the_untrusted_binary(self):
        with patch.object(doctor, "bundled_tool", side_effect=RuntimeError("Tool checksum differs")):
            report = doctor.collect_report(self.root)
        self.assertFalse(report["ready"])
        self.assertTrue(all(not item["integrity_verified"] for item in report["tools"] if item["required"]))
        self.assertTrue(all(call.args[0][0] == "nvidia-smi" for call in self.mocks[-1].call_args_list))

    def test_report_has_no_personal_paths_raw_errors_or_gpu_identifiers(self):
        secret = str(self.root / "PRIVATE_RECORDING.mp4")
        def output(command):
            if command[0] == "nvidia-smi":
                return subprocess.CompletedProcess(command, 0, secret + ", 581.42, 8192\n", "UUID-PRIVATE")
            return subprocess.CompletedProcess(command, 1, secret, "C:\\Users\\PRIVATE_NAME\\failure")
        with patch.object(doctor, "_execute", side_effect=output):
            serialized = json.dumps(doctor.collect_report(self.root))
        for forbidden in (str(self.root), "PRIVATE_RECORDING", "PRIVATE_NAME", "UUID-PRIVATE", "private-personal-workspace"):
            self.assertNotIn(forbidden, serialized)

    def test_present_model_is_never_loaded_hashed_or_claimed_validated(self):
        model = self.root / "models/qwen3-vl-2b"
        model.mkdir(parents=True)
        (model / "weights.bin").write_bytes(b"PRIVATE MODEL CONTENT")
        self.save(model / "manifest.json", {"assets": [{"file": "weights.bin", "size_bytes": 21}]})
        (self.root / ".venv-ai/Scripts").mkdir(parents=True)
        (self.root / ".venv-ai/Scripts/python.exe").touch()
        with patch.object(Path, "read_bytes", side_effect=AssertionError("No model content reads")), \
             patch("hashlib.file_digest", side_effect=AssertionError("No model hashes; tool verification is mocked")):
            report = doctor.collect_report(self.root)
        video = report["optional_models"][0]
        self.assertTrue(video["files_present"])
        self.assertEqual(video["integrity"], "not checked by doctor")
        self.assertEqual(video["inference"], "not run")
        self.assertNotIn("PRIVATE MODEL CONTENT", json.dumps(report))

    def test_output_only_replaces_a_prior_readiness_report(self):
        report = doctor.collect_report(self.root)
        target = self.root / "logs/readiness.json"
        doctor.save_report(target, report)
        doctor.save_report(target, report)
        self.assertEqual(json.loads(target.read_text())["report_kind"], "fpv-sesh-readiness")
        for existing in ({"shots": []}, [], None):
            self.save(target, existing)
            before = target.read_bytes()
            with self.assertRaises(ValueError):
                doctor.save_report(target, report)
            self.assertEqual(target.read_bytes(), before)
        with self.assertRaises(ValueError):
            doctor.save_report(self.root / "original.mp4", report)
        self.assertFalse((self.root / "original.mp4").exists())

    def test_json_cli_exit_status_and_explicit_report(self):
        report = doctor.collect_report(self.root)
        output = self.root / "logs/shareable.json"
        with patch.object(doctor, "collect_report", return_value=report), contextlib.redirect_stdout(io.StringIO()) as stream:
            result = doctor.main(["--json", "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(stream.getvalue())["ready"])
        self.assertTrue(output.is_file())
        with patch.object(doctor, "collect_report", return_value={**report, "ready": False}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(doctor.main(["--json"]), 1)


if __name__ == "__main__":
    unittest.main()
