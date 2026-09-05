"""Current-tool selection and verified fallback when a GPU encoder is unavailable."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fpvsesh import enhance, media


class EnhancementTests(unittest.TestCase):
    def test_default_tools_follow_current_media_selection(self):
        with patch.object(enhance, "locate_tools", return_value=("current/ffmpeg.exe", "current/ffprobe.exe")) as locate:
            self.assertEqual(enhance._selected_tools(), (Path("current/ffmpeg.exe"), Path("current/ffprobe.exe")))
            locate.assert_called_once_with()
        with patch.object(enhance, "locate_tools", side_effect=AssertionError("Explicit tools must be respected")):
            self.assertEqual(enhance._selected_tools("custom/ffmpeg.exe"),
                             (Path("custom/ffmpeg.exe"), Path("custom/ffprobe.exe")))

    def test_gpu_encoder_failure_falls_back_to_verified_cpu_without_changing_source(self):
        with tempfile.TemporaryDirectory(prefix="fpvsesh-enhancement-fallback-") as directory:
            folder = Path(directory)
            source = folder / "source café [original].mp4"
            target = folder / "result.mp4"
            ffmpeg, ffprobe = media.locate_tools()
            media.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=60",
                       "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", str(source)])
            original = media.sha256_file(source)
            original_run = enhance._run
            def unavailable_gpu(command, name, log, timeout):
                if "nvenc" in name:
                    raise RuntimeError("GPU encoder unavailable with this driver")
                return original_run(command, name, log, timeout)
            with patch.object(enhance, "ROOT", folder), patch.object(enhance, "_run", side_effect=unavailable_gpu):
                record = enhance.enhance_segment(source, target,
                    {"quality": "auto", "codec": "hevc", "start": .2, "duration": .3,
                     "fps": "60/1", "frames": 18, "width": 320, "height": 240,
                     "ffmpeg": ffmpeg, "ffprobe": ffprobe}, lambda message: None)
            self.assertEqual((record["backend"], record["encoder"]), ("lanczos-cpu", "libx265"))
            self.assertFalse(record["ai_inference"])
            self.assertEqual(len(record["warnings"]), 2)
            self.assertTrue(all("GPU encoder unavailable" in warning for warning in record["warnings"]))
            checked = media.validate_output(target, 18, "60/1", 320, 240)
            self.assertTrue(checked["passed"], checked["errors"])
            self.assertEqual(media.sha256_file(source), original)
            self.assertFalse(list(folder.glob("*.working-*")))


if __name__ == "__main__":
    unittest.main()
