"""Frame-boundary control and worker cleanup, without importing CUDA or torch."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fpvsesh.control import Cancelled, check_control


def write_control(path, action):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"action": action}), encoding="utf-8")
    temporary.replace(path)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "control.json"

    def test_missing_and_resume_controls_continue_without_deleting_commands(self):
        check_control(self.path)
        write_control(self.path, "resume")
        check_control(self.path)
        self.assertEqual(json.loads(self.path.read_text())["action"], "resume")
        with self.assertRaises(ValueError):
            check_control(self.path, poll_interval=0)

    def test_pause_waits_through_partial_write_then_resumes_once(self):
        write_control(self.path, "pause")
        paused, finished = threading.Event(), threading.Event()
        resumed = Mock()
        def wait():
            check_control(self.path, on_pause=paused.set, on_resume=resumed, poll_interval=.01)
            finished.set()
        thread = threading.Thread(target=wait, daemon=True)
        thread.start()
        try:
            self.assertTrue(paused.wait(1))
            self.assertFalse(finished.is_set())
            self.path.write_text("{", encoding="utf-8")
            self.assertFalse(finished.wait(.04), "An incomplete UI write must not resume paused work")
            write_control(self.path, "resume")
            self.assertTrue(finished.wait(1))
            resumed.assert_called_once_with()
        finally:
            self.path.unlink(missing_ok=True)
            thread.join(1)

    def test_cancel_while_paused_raises_shared_exception_and_preserves_command(self):
        write_control(self.path, "pause")
        paused, finished = threading.Event(), threading.Event()
        result = []
        def wait():
            try:
                check_control(self.path, on_pause=paused.set, poll_interval=.01)
            except Exception as error:
                result.append(error)
            finally:
                finished.set()
        thread = threading.Thread(target=wait, daemon=True)
        thread.start()
        try:
            self.assertTrue(paused.wait(1))
            write_control(self.path, "cancel")
            self.assertTrue(finished.wait(1))
            self.assertIsInstance(result[0], Cancelled)
            self.assertEqual(json.loads(self.path.read_text())["action"], "cancel")
        finally:
            self.path.unlink(missing_ok=True)
            thread.join(1)


class RecordingPipe(io.BytesIO):
    def __init__(self, *args):
        super().__init__(*args)
        self.recorded = None
        self.wrote = threading.Event()

    def write(self, value):
        count = super().write(value)
        self.wrote.set()
        return count

    def close(self):
        if not self.closed:
            self.recorded = self.getvalue()
        super().close()


class FakeProcess:
    def __init__(self, *, stdin=None, stdout=None):
        self.stdin, self.stdout = stdin, stdout
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -1

    def kill(self):
        self.terminate()


class WorkerControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the actual worker with only its model dependency substituted.
        # Commands, frame reads/writes, controls, and cleanup remain real code.
        fake_models = ModuleType("fpvsesh.ai_models")
        fake_models.Restorer = Mock()
        path = Path(__file__).resolve().parents[1] / "fpvsesh/ai_worker.py"
        spec = importlib.util.spec_from_file_location("fpvsesh._tested_control_worker", path)
        cls.worker = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"fpvsesh.ai_models": fake_models}):
            spec.loader.exec_module(cls.worker)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "cache").mkdir()
        self.control = self.root / "cache/control.json"
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"untouched original")
        self.output = self.root / "cache/selected.mp4"
        self.config = {"source": str(self.source), "output": str(self.output), "ffmpeg": "mock-ffmpeg",
                       "frames": 3, "fps": "60/1", "start": 2,
                       "source_width": 2, "source_height": 2,
                       "content_width": 4, "content_height": 4, "width": 8, "height": 4,
                       "ai_model": "RealESRGAN_x2plus", "ai_blend": 1}
        self.frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (20, 80, 140)]
        self.decoder = FakeProcess(stdout=RecordingPipe(b"".join(frame.tobytes() for frame in self.frames)))
        self.encoder = FakeProcess(stdin=RecordingPipe())
        self.calls = []
        self.on_frame = None
        def enhance(original, outscale):
            self.calls.append(original.copy())
            if self.on_frame:
                self.on_frame(len(self.calls))
            return np.repeat(np.repeat(original, 2, axis=0), 2, axis=1)
        self.restorer = SimpleNamespace(enhance=enhance, close=Mock())
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(self.worker, "ROOT", self.root))
        self.constructor = self.stack.enter_context(patch.object(self.worker, "Restorer", return_value=self.restorer))
        self.popen = self.stack.enter_context(patch.object(self.worker.subprocess, "Popen", side_effect=[self.decoder, self.encoder]))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def assert_cleanup(self):
        self.assertIsNotNone(self.decoder.poll())
        self.assertIsNotNone(self.encoder.poll())
        self.assertTrue(self.decoder.stdout.closed)
        self.assertTrue(self.encoder.stdin.closed)
        self.restorer.close.assert_called_once_with()
        self.assertEqual(self.source.read_bytes(), b"untouched original")

    def test_cancel_during_inference_finishes_one_frame_then_cleans_up(self):
        self.on_frame = lambda count: write_control(self.control, "cancel")
        with self.assertRaises(Cancelled):
            self.worker.render(self.config)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.encoder.stdin.recorded, bytes([20]) * (4 * 4 * 3))
        self.assertFalse(self.output.with_suffix(".mp4.ai.json").exists())
        self.assertTrue(self.encoder.terminated)
        self.assert_cleanup()

    def test_pause_after_frame_and_resume_retains_order_and_all_frame_bytes(self):
        self.on_frame = lambda count: write_control(self.control, "pause") if count == 1 else None
        result, errors = [], []
        def render():
            try:
                result.append(self.worker.render(self.config))
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=render, daemon=True)
        thread.start()
        try:
            self.assertTrue(self.encoder.stdin.wrote.wait(1))
            time.sleep(.04)
            self.assertEqual(len(self.calls), 1)
            self.assertFalse(result)
            write_control(self.control, "resume")
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors)
            self.assertEqual(result[0]["frames"], 3)
            self.assertEqual(self.encoder.stdin.recorded, b"".join(bytes([v]) * 48 for v in (20, 80, 140)))
            self.assertTrue(self.output.with_suffix(".mp4.ai.json").exists())
            self.assert_cleanup()
        finally:
            self.control.unlink(missing_ok=True)
            thread.join(2)

    def test_stopped_encoder_pipe_cannot_mask_cancellation(self):
        self.on_frame = lambda count: write_control(self.control, "cancel")
        close = self.encoder.stdin.close
        def stopped_pipe():
            close()
            raise BrokenPipeError("Encoder stopped before buffered pipe close")
        with patch.object(self.encoder.stdin, "close", side_effect=stopped_pipe):
            with self.assertRaises(Cancelled):
                self.worker.render(self.config)
        self.assert_cleanup()

    def test_cancel_before_setup_avoids_model_and_process_work(self):
        write_control(self.control, "cancel")
        with self.assertRaises(Cancelled):
            self.worker.render(self.config)
        self.constructor.assert_not_called()
        self.popen.assert_not_called()

    def test_short_decode_fails_without_success_record_and_releases_processes(self):
        self.decoder.stdout = RecordingPipe(b"short")
        with self.assertRaisesRegex(RuntimeError, "5 of 12 frame bytes"):
            self.worker.render(self.config)
        self.assertEqual(len(self.calls), 0)
        self.assertFalse(self.output.with_suffix(".mp4.ai.json").exists())
        self.assert_cleanup()

    def test_main_returns_distinct_cancel_exit_code(self):
        config = self.root / "config.json"
        config.write_text("{}", encoding="utf-8")
        with patch.object(sys, "argv", ["worker", "--config", str(config)]), \
                patch.object(self.worker, "render", side_effect=Cancelled("stopped")):
            self.assertEqual(self.worker.main(), 75)


if __name__ == "__main__":
    unittest.main()
