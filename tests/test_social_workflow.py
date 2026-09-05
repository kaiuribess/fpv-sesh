"""Combined real-media CLI music/social/recovery coverage in an isolated copy."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from fpvsesh import media
import test_cli


class MusicSocialWorkflow(unittest.TestCase):
    setUpClass = classmethod(test_cli.CliEndToEndTests.setUpClass.__func__)
    tearDownClass = classmethod(test_cli.CliEndToEndTests.tearDownClass.__func__)
    run_cli = test_cli.CliEndToEndTests.run_cli

    def test_generated_audio_selected_as_music_is_rejected_before_any_overwrite(self):
        job = self.app / "output" / "music-collision"
        job.mkdir()
        original = job / "source-audio.wav"
        media.run([self.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000",
                   "-t", "0.25", "-c:a", "pcm_s16le", str(original)])
        digest = media.sha256_file(original)
        env = {**os.environ, "FPVSESH_FFMPEG": self.ffmpeg, "FPVSESH_FFPROBE": self.ffprobe}
        env.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-m", "fpvsesh.cli", "make", "--input", str(self.paths[0]),
                                 "--job", str(job), "--preview-only", "--music", str(original)],
                                cwd=self.app, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("original music file outside", result.stdout)
        self.assertEqual(media.sha256_file(original), digest)
        self.assertFalse((job / "audio").exists())
        self.assertFalse((job / "music-analysis.json").exists())
        self.assertFalse((job / "preview.mp4").exists())

    def test_music_social_pack_then_remove_music_keeps_video_decisions_and_sources(self):
        music = self.folder / "music café & sample.wav"
        media.run([self.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=660:sample_rate=48000",
                   "-t", "2", "-c:a", "pcm_s16le", str(music)])
        music_hash = media.sha256_file(music)
        job = self.app / "output" / "music-social"
        args = ["make", "--input", str(self.paths[0]), "--input", str(self.paths[1]), "--job", str(job),
                "--preview-only", "--music", str(music), "--music-offset", ".25", "--music-end", "loop",
                "--music-level", ".65", "--social-formats", "vertical,square,portrait", "--framing", "blur"]
        events, _, _ = self.run_cli(args)
        self.assertEqual(events[-1]["stage"], "complete")
        timeline = json.loads((job / "timeline.json").read_text(encoding="utf-8"))
        settings = json.loads((job / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(settings["music"], str(music.resolve()))
        self.assertEqual(settings["social_formats"], ["vertical", "square", "portrait"])
        self.assertTrue((job / "source-audio.wav").is_file())
        mix = json.loads((job / "music-mix.json").read_text())
        self.assertTrue(mix["verification"]["decoded_duration"] > 2)
        for code, size in (("vertical", (360, 640)), ("square", (360, 360)), ("portrait", (360, 450))):
            result = media.probe(job / "social-preview" / (code + ".mp4"))
            self.assertEqual((result["width"], result["height"]), size)
            self.assertEqual(result["frame_count"], timeline["frames"])
            self.assertTrue(result["audio"])
        self.assertTrue((job / "edit.csv").is_file())
        self.assertEqual(media.sha256_file(music), music_hash)
        self.assertEqual([media.sha256_file(path) for path in self.paths], self.hashes)
        # Cache reuse must not change the job fingerprint or archive its outputs.
        fingerprint = json.loads((job / "artifact-state.json").read_text())["fingerprint"]
        events, _, _ = self.run_cli(["make", "--job", str(job), "--preview-only"])
        self.assertEqual(events[-1]["stage"], "complete")
        self.assertEqual(json.loads((job / "artifact-state.json").read_text())["fingerprint"], fingerprint)
        self.assertFalse((job / "preview.previous.mp4").exists())
        events, _, _ = self.run_cli(["make", "--job", str(job), "--preview-only", "--no-music", "--social-formats", "none"])
        self.assertEqual(events[-1]["stage"], "complete")
        self.assertIsNone(json.loads((job / "settings.json").read_text(encoding="utf-8"))["music"])
        self.assertFalse((job / "music-mix.json").exists())
        self.assertFalse((job / "social-preview/vertical.mp4").exists())
        self.assertTrue((job / "social-preview/vertical.previous.mp4").is_file())


if __name__ == "__main__":
    unittest.main()
