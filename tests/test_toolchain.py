import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fpvsesh import media, toolchain
from fpvsesh.installation import locked_versions
from scripts.build_release import safe_release_path


class ToolchainTests(unittest.TestCase):
    def test_current_manifest_pair_ignores_other_downloads_and_rechecks_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            current = root / "tools/current"
            current.mkdir(parents=True)
            record = {"name": "FFmpeg", "required": True, "executable_sha256": {}}
            for kind, field in (("ffmpeg", "executable"), ("ffprobe", "probe_executable")):
                path = current / (kind + ".exe")
                path.write_bytes(kind.encode())
                record[field] = path.relative_to(root).as_posix()
                record["executable_sha256"][kind] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "tools/dependencies.json").write_text(json.dumps({"tools": [record]}))
            (root / "tools/ffmpeg.exe").write_bytes(b"unrelated download")
            with patch.object(media, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(media.locate_tools(), (str(current / "ffmpeg.exe"), str(current / "ffprobe.exe")))
                (current / "ffmpeg.exe").write_bytes(b"changed executable")
                with self.assertRaisesRegex(RuntimeError, "integrity check"):
                    media.locate_tools()

    def test_override_requires_a_deliberate_complete_pair(self):
        with patch.dict(os.environ, {"FPVSESH_FFMPEG": "missing.exe"}, clear=True):
            with self.assertRaisesRegex(ValueError, "both"):
                media.locate_tools()

    def test_release_excludes_user_content_and_downloaded_code(self):
        for name in ("input/clip.mp4", "logs/report.json", "output/job/settings.json", "models/weights.pth",
                     "tools/bin/ffmpeg.exe", ".env", ".env.production", ".venv/pyvenv.cfg", "../outside.py",
                     "Original.M4V", "track.m4a", ".ENV.PRIVATE", "Output/settings.json"):
            with self.subTest(name=name):
                self.assertFalse(safe_release_path(name))
        for name in ("fpvsesh/ui.py", "models/qwen3-vl-2b/manifest.json", "docs/user-guide.md"):
            self.assertTrue(safe_release_path(name))

    def test_version_check_understands_hash_locked_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.txt"
            path.write_text("# comment\npackage-one==1.2 --hash=sha256:abc --hash=sha256:def\n")
            self.assertEqual(locked_versions(path), {"package-one": "1.2"})
