"""Failed source preflight must not overwrite a previously completed edit."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fpvsesh import cli


class PreflightPreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fpvsesh-preflight-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.job = self.root / "output" / "completed"
        self.job.mkdir(parents=True)
        self.source = self.root / "new source.mp4"
        self.source.write_bytes(b"mock source; never rendered")
        self.source_data = {"source": str(self.source), "sha256": "a" * 64, "filename": self.source.name,
                            "sample_aspect_ratio": "1:1", "hdr": False, "color_space": "bt709",
                            "color_transfer": "bt709", "color_primaries": "bt709", "color_range": "tv"}
        protected = {
            "settings.json": {"style": "flow", "music": "previous soundtrack.wav", "inputs": ["old source.mp4"]},
            "sources.json": [{"source": "old source.mp4", "sha256": "b" * 64}],
            "music-analysis.json": {"path": "previous soundtrack.wav", "duration": 100},
            "music-mix.json": {"old": "verified mix"},
            "timeline.json": {"shots": [{"start": 2.4, "end": 10.7}]},
            "exports.json": {"master": str(self.job / "final_4k.mp4")},
            "artifact-state.json": {"fingerprint": "previous-valid-render"},
        }
        for name, value in protected.items():
            (self.job / name).write_text(json.dumps(value), encoding="utf-8")
        (self.job / "final_4k.mp4").write_bytes(b"untouched completed master")
        (self.job / "preview.mp4").write_bytes(b"untouched completed preview")
        self.before = {path.name: path.read_bytes() for path in self.job.iterdir()}

    def invoke(self, source, scan=None, job=None):
        args = cli.parser().parse_args(["make", "--job", str(job or self.job), "--input", str(self.source),
                                       "--style", "hype", "--no-music", "--analyze-only"])
        with (patch.object(cli, "ROOT", self.root), patch.object(cli, "probe", return_value=source),
              patch.object(cli, "hardware_diagnostics", return_value={}),
              patch("fpvsesh.enhance.backend_status", return_value={}),
              patch.object(cli, "ensure_space"), patch.object(cli, "inspect_timestamps", return_value=scan),
              redirect_stdout(io.StringIO())):
            cli.make(args)

    def assert_preserved(self):
        for name, content in self.before.items():
            self.assertEqual((self.job / name).read_bytes(), content, name)
        self.assertFalse((self.job / "final_4k.previous.mp4").exists())

    def test_hdr_rejection_preserves_saved_settings_music_and_exports(self):
        with self.assertRaisesRegex(ValueError, "HDR input"):
            self.invoke(dict(self.source_data, hdr=True))
        self.assert_preserved()

    def test_timestamp_rejection_preserves_saved_settings_music_and_exports(self):
        scan = {"scan_complete": False, "strictly_monotonic": True, "decode_error_count": 0}
        with self.assertRaisesRegex(ValueError, "decode/timestamp problems"):
            self.invoke(self.source_data, scan)
        self.assert_preserved()

    def test_output_root_is_not_a_job(self):
        with self.assertRaisesRegex(ValueError, "not the output folder itself"):
            self.invoke(self.source_data, job=self.root / "output")
        self.assert_preserved()
        self.assertFalse((self.root / "output" / "settings.json").exists())


if __name__ == "__main__":
    unittest.main()
