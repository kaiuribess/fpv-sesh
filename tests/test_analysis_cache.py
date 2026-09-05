"""Full-source cache integrity and fractional-duration proxy regressions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fpvsesh import analysis, media


class AnalysisCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fpvsesh-analysis-integrity-")
        cls.folder = Path(cls.temporary.name)
        cls.ffmpeg, _ = media.locate_tools()
        cls.source = cls.folder / "flight's [real] café 🚁.mp4"
        # This even-sized input previously scaled to an odd 480x273 proxy.
        media.run([cls.ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                   "testsrc2=size=640x364:rate=60000/1001", "-frames:v", "91",
                   "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(cls.source)])
        cls.probe = media.probe(cls.source)
        cls.original_hash = media.sha256_file(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_analysis(self, cache, source=None):
        events = []
        result = analysis.analyze(source or self.probe, cache, lambda *args: events.append(args))
        return result, events

    def test_full_fractional_duration_is_cached_and_source_is_unchanged(self):
        cache = self.folder / "complete-cache"
        first, _ = self.run_analysis(cache)
        self.assertEqual(first["proxy_frames"], 19)
        self.assertEqual(len(first["rows"]), 10)
        self.assertGreaterEqual(first["coverage_seconds"], self.probe["duration"])
        second, events = self.run_analysis(cache)
        self.assertEqual(first, second)
        self.assertTrue(any("Reusing full-session" in event[2] for event in events))
        self.assertFalse(any(event[0] == "proxy" for event in events))
        self.assertEqual(media.sha256_file(self.source), self.original_hash)

    def test_truncated_rows_and_modified_proxy_cannot_be_reused(self):
        cache = self.folder / "damaged-cache"
        original, _ = self.run_analysis(cache)
        proxy = Path(original["proxy"])
        record = proxy.with_name("analysis.json")
        truncated = json.loads(record.read_text(encoding="utf-8"))
        truncated["rows"] = truncated["rows"][:3]
        record.write_text(json.dumps(truncated), encoding="utf-8")
        repaired, events = self.run_analysis(cache)
        self.assertEqual(len(repaired["rows"]), 10)
        self.assertTrue(any(event[0] == "proxy" for event in events))
        # A syntactically valid completed record must not hide replacement or
        # partial proxy bytes left by a crash, transfer, or disk corruption.
        proxy.write_bytes(b"not a complete MP4")
        repaired, events = self.run_analysis(cache)
        self.assertTrue(any(event[0] == "proxy" for event in events))
        self.assertEqual(media.sha256_file(proxy), repaired["proxy_sha256"])

    def test_clean_early_eof_cannot_claim_the_longer_requested_source(self):
        cache = self.folder / "early-eof-cache"
        incorrect = dict(self.probe, duration=self.probe["duration"] + 2)
        with self.assertRaisesRegex(RuntimeError, "Proxy decode incomplete"):
            self.run_analysis(cache, incorrect)
        key = hashlib.sha256((str(analysis.identity(incorrect)) + analysis.ANALYSIS_VERSION).encode()).hexdigest()[:20]
        self.assertFalse((cache / "analysis" / key / "analysis.json").exists())
        self.assertEqual(media.sha256_file(self.source), self.original_hash)


if __name__ == "__main__":
    unittest.main()
