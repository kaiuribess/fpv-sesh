"""Source playback arguments and provenance, without opening a visible player."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fpvsesh import source_review


class SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="source-review-")
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "飞行 & $HOME's [clip].mp4"
        self.source.write_bytes(b"untouched original recording")
        self.player = self.root / "tools/ffmpeg-7.1.1/bin/ffplay.exe"
        self.player.parent.mkdir(parents=True)
        self.player.write_bytes(b"mock verified player")
        self.record = {"name": "FFmpeg", "sha256": source_review.FFMPEG_ARCHIVE_SHA256,
                       "executable": "tools/ffmpeg-7.1.1/bin/ffmpeg.exe"}
        (self.root / "tools/dependencies.json").write_text(json.dumps({"tools": [self.record]}))
        self.probe_patch = patch.object(source_review, "probe", return_value={"duration": 20.0})
        self.probe = self.probe_patch.start()
        self.hash_patch = patch.object(source_review, "FFPLAY_SHA256", hashlib.sha256(self.player.read_bytes()).hexdigest())
        self.hash_patch.start()

    def tearDown(self):
        self.hash_patch.stop()
        self.probe_patch.stop()
        self.temp.cleanup()

    def test_context_is_clamped_to_current_source_and_never_trusts_stale_duration(self):
        self.assertEqual(source_review.normalize_section(self.source, 1, 4)["start"], 0)
        middle = source_review.normalize_section(self.source, 6, 10, source_duration=200)
        self.assertEqual((middle["start"], middle["end"], middle["duration"]), (4, 12, 8))
        tail = source_review.normalize_section(self.source, 17, 19.5)
        self.assertEqual((tail["start"], tail["end"]), (15, 20))
        self.probe.assert_called_with(self.source, include_hash=False)
        with self.assertRaisesRegex(ValueError, "beyond the current recording"):
            source_review.normalize_section(self.source, 17, 30, source_duration=200)

    def test_verified_bundled_player_only_and_integrity_failure(self):
        self.assertEqual(source_review.find_player(self.root), self.player)
        self.player.write_bytes(b"unexpected replacement")
        with self.assertRaisesRegex(RuntimeError, "integrity check"):
            source_review.find_player(self.root)

    def test_missing_or_untrusted_player_never_uses_path_fallback(self):
        self.record["sha256"] = "0"*64
        (self.root / "tools/dependencies.json").write_text(json.dumps({"tools": [self.record]}))
        with self.assertRaisesRegex(FileNotFoundError, "bundled source player"):
            source_review.find_player(self.root)
        for manifest in ([], {"tools": None}, {"tools": [None]}):
            (self.root / "tools/dependencies.json").write_text(json.dumps(manifest))
            with self.assertRaises(FileNotFoundError):
                source_review.find_player(self.root)

    def test_structured_cpu_player_launch_preserves_files_and_environment(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        environment = dict(os.environ)
        with patch.object(source_review.subprocess, "Popen", return_value=MagicMock(pid=1234)) as spawn:
            result = source_review.play_section(self.source, 5, 11, app_dir=self.root)
        self.assertEqual((result["start"], result["end"], result["pid"]), (3, 13, 1234))
        command = spawn.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertEqual(command[0], str(self.player))
        self.assertEqual(command[-2:], ["-i", str(self.source)])
        self.assertEqual(command[command.index("-ss")+1], "3.000000")
        self.assertEqual(command[command.index("-t")+1], "10.000000")
        self.assertEqual(command[command.index("-threads")+1], "2")
        self.assertNotIn("-enable_vulkan", command)
        self.assertNotIn("-hwaccel", command)
        self.assertFalse(spawn.call_args.kwargs["shell"])
        self.assertEqual(spawn.call_args.kwargs["env"]["SDL_RENDER_DRIVER"], "software")
        self.assertEqual(spawn.call_args.kwargs["creationflags"], getattr(source_review.subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(dict(os.environ), environment)
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, before)

    def test_invalid_bounds_and_missing_source_cannot_launch(self):
        with patch.object(source_review.subprocess, "Popen") as spawn:
            for start, end in ((-1, 1), (1, 1), (True, 2), (0, math.nan), (0, math.inf)):
                with self.assertRaises(ValueError):
                    source_review.play_section(self.source, start, end, app_dir=self.root)
            with self.assertRaises(FileNotFoundError):
                source_review.play_section(self.root / "missing.mp4", 1, 2, app_dir=self.root)
            with self.assertRaises(ValueError):
                source_review.play_section(self.source, 1, 2, context=-1, app_dir=self.root)
            spawn.assert_not_called()

    def test_launch_failure_is_clear(self):
        with patch.object(source_review.subprocess, "Popen", side_effect=OSError("player failed")), \
                self.assertRaisesRegex(RuntimeError, "source player could not open"):
            source_review.play_section(self.source, 1, 3, app_dir=self.root)


if __name__ == "__main__":
    unittest.main()
