"""Recognition-only refresh preserves source identities and completed edits."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from fpvsesh import cli, video_review


class VideoReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fpvsesh-recognition-review-")
        self.root = (Path(self.temp.name) / "application").resolve()
        self.job = self.root / "output" / "Reviewed session"
        self.job.mkdir(parents=True)
        (self.root / "cache").mkdir()
        self.source_paths, self.sources = [], []
        for index in range(2):
            source = (Path(self.temp.name) / f"Original 飞行 {index}.mp4").resolve()
            source.write_bytes(f"read-only original {index}".encode())
            self.source_paths.append(source)
            stat = source.stat()
            self.sources.append({"source": str(source), "filename": source.name, "duration": 12.0,
                                 "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                 "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        self.candidates = [{"id": "kept-original", "source": self.sources[0]["source"],
                            "identity": self.sources[0]["sha256"], "start": 1.25, "end": 8.75,
                            "score": 87.5, "selected": True, "review_key": "whole-trick",
                            "flight_label": "old estimate", "trick_old": "stale"}]
        self.write("sources.json", self.sources)
        self.write("settings.json", {"recognition": "off", "quality": "ai", "music": "music.wav", "recovery": 3.5})
        self.write("candidates.json", {"candidates": self.candidates, "overrides": {"keep": ["kept-original"]}, "coverage": "saved"})
        self.write("flight-map.json", {"sources": [], "prior": "reviewed map"})
        self.write("timeline.json", {"shots": self.candidates, "fps": "60000/1001", "duration": 7.5})
        self.write("status.json", {"stage": "complete", "message": "Rendered original edit"})
        self.write("artifact-state.json", {"fingerprint": "reviewed-render-fingerprint"})
        self.write("exports.json", {"master": "final_4k.mp4"})
        (self.job / "events.jsonl").write_bytes(b'{"stage":"complete"}\n')
        (self.job / "final_4k.mp4").write_bytes(b"existing rendered master bytes")
        (self.job / "preview.mp4").write_bytes(b"existing preview bytes")
        (self.job / "social").mkdir()
        (self.job / "social" / "vertical.mp4").write_bytes(b"existing vertical export bytes")
        self.before = self.snapshot()
        self.patchers = [patch.object(video_review, "ROOT", self.root),
                         patch.object(video_review, "probe", side_effect=self.fake_probe),
                         patch.object(video_review, "analyze", side_effect=self.fake_analysis),
                         patch.object(video_review, "build_flight_map", return_value={"sources": [], "new": "video recognition",
                             "learning": {"video_model": {"available": True, "coverage_complete": True,
                                 "windows_analyzed": 4, "windows_requested": 4, "coverage_seconds": 24}}}),
                         patch.object(video_review, "annotate_candidates", side_effect=self.fake_annotations)]
        self.mocks = [patcher.start() for patcher in self.patchers]
        self.probe, self.analyze, self.build, self.annotate = self.mocks[1:]

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def write(self, name, value):
        (self.job / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    def snapshot(self):
        return {str(path.relative_to(self.job)): path.read_bytes() for path in self.job.rglob("*")
                if path.is_file() and path.name != "recognition-status.json"}

    def fake_probe(self, path, *, include_hash):
        self.assertTrue(include_hash)
        path = Path(path)
        record = copy.deepcopy(next(source for source in self.sources if source["source"] == str(path)))
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record.update(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        return record

    def fake_analysis(self, source, cache, event, checkpoint):
        checkpoint()
        return {"source": source["source"], "identity": source["sha256"], "duration": 12, "rows": []}

    def fake_annotations(self, candidates, analyses, learning):
        candidates[0].update(flight_label="ordinary flight", trick_label="uncertain", scene_context={"label": "park"})
        # Even an accidental annotation implementation regression cannot retime
        # or select a shot through this recognition-only operation.
        candidates[0].update(start=0, score=999, selected=False, id="changed")
        return candidates

    def run_review(self, mode="thorough"):
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = video_review.map_flight(self.job, mode)
        self.events = [json.loads(line) for line in stream.getvalue().splitlines()]
        return result

    def test_refresh_changes_only_annotations_and_recognition_setting(self):
        result = self.run_review()
        self.assertFalse(result["cancelled"])
        after = self.snapshot()
        permitted = {"flight-map.json", "candidates.json", "settings.json"}
        self.assertEqual(set(after), set(self.before))
        for name in self.before:
            if name not in permitted:
                self.assertEqual(after[name], self.before[name], name)
        saved = json.loads(after["candidates.json"])
        self.assertEqual(saved["overrides"], {"keep": ["kept-original"]})
        candidate = saved["candidates"][0]
        for key, value in self.candidates[0].items():
            if not key.startswith(video_review.ANNOTATION_PREFIXES):
                self.assertEqual(candidate[key], value, key)
        self.assertEqual(candidate["trick_label"], "uncertain")
        self.assertNotIn("trick_old", candidate)
        settings = json.loads(after["settings.json"])
        self.assertEqual(settings, {"recognition": "thorough", "quality": "ai", "music": "music.wav", "recovery": 3.5})
        self.assertEqual(self.build.call_args.kwargs["recognition"], "thorough")
        self.assertTrue(all(event["operation"] == "map-flight" for event in self.events))
        self.assertEqual(self.events[-1]["stage"], "complete")
        self.assertEqual(len(self.probe.call_args_list), 2)
        for source, path in zip(self.sources, self.source_paths):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["sha256"])

    def test_every_source_hash_is_checked_before_any_analysis_or_annotation(self):
        self.source_paths[1].write_bytes(b"replacement recording")
        with self.assertRaisesRegex(ValueError, "Recording changed"):
            self.run_review()
        self.assertEqual(self.snapshot(), self.before)
        self.analyze.assert_not_called()
        self.build.assert_not_called()
        self.annotate.assert_not_called()

    def test_cancel_during_model_preserves_existing_job(self):
        def cancel(*args, **kwargs):
            (self.root / "cache/control.json").write_text('{"action":"cancel"}')
            args[4]()
        self.build.side_effect = cancel
        result = self.run_review()
        self.assertTrue(result["cancelled"])
        self.assertEqual(self.snapshot(), self.before)
        self.assertEqual(self.events[-1]["stage"], "cancelled")
        self.annotate.assert_not_called()

    def test_cancel_after_annotations_still_commits_nothing(self):
        def cancel(candidates, analyses, learning):
            self.fake_annotations(candidates, analyses, learning)
            (self.root / "cache/control.json").write_text('{"action":"cancel"}')
        self.annotate.side_effect = cancel
        self.assertTrue(self.run_review()["cancelled"])
        self.assertEqual(self.snapshot(), self.before)

    def test_source_changed_during_model_is_not_committed(self):
        def change(*args, **kwargs):
            self.source_paths[0].write_bytes(b"externally edited while reviewing")
            return {"new": "map"}
        self.build.side_effect = change
        with self.assertRaisesRegex(RuntimeError, "recording changed"):
            self.run_review()
        self.assertEqual(self.snapshot(), self.before)

    def test_concurrent_saved_edits_are_not_overwritten(self):
        def edit(*args, **kwargs):
            self.write("settings.json", {"external": "user changed this"})
            return {"new": "map"}
        self.build.side_effect = edit
        with self.assertRaisesRegex(RuntimeError, "settings.json changed"):
            self.run_review()
        after = self.snapshot()
        self.assertEqual(json.loads(after["settings.json"]), {"external": "user changed this"})
        for name, content in self.before.items():
            if name != "settings.json":
                self.assertEqual(after[name], content)

    def test_write_failure_rolls_back_already_replaced_annotations(self):
        replace = video_review._replace_bytes
        def fail_once(path, content):
            if path.name == "settings.json":
                raise OSError("intentional write failure")
            return replace(path, content)
        with patch.object(video_review, "_replace_bytes", side_effect=fail_once), self.assertRaisesRegex(OSError, "intentional"):
            self.run_review()
        self.assertEqual(self.snapshot(), self.before)
        self.assertFalse(list(self.job.glob(".recognition-*.tmp")))

    def test_stale_candidate_and_invalid_hash_fail_before_model(self):
        self.candidates[0]["identity"] = "a" * 64
        self.write("candidates.json", {"candidates": self.candidates})
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "stale source identity"):
            self.run_review()
        self.assertEqual(self.snapshot(), before)
        self.build.assert_not_called()
        self.sources[0]["sha256"] = "not a content hash"
        self.write("sources.json", self.sources)
        with self.assertRaisesRegex(ValueError, "complete SHA256"):
            self.run_review()

    def test_job_must_be_existing_output_child_and_lock_prevents_overlap(self):
        for path in (self.root, self.root / "output", Path(self.temp.name)):
            with self.assertRaisesRegex(ValueError, "existing job folder"):
                video_review.map_flight(path)
        with video_review._run_lock(self.root / "cache"), self.assertRaisesRegex(RuntimeError, "Another FPV Sesh"):
            self.run_review()
        self.assertEqual(self.snapshot(), self.before)

    def test_off_mode_and_cli_contract(self):
        self.run_review("off")
        self.assertEqual(self.build.call_args.kwargs["recognition"], "off")
        args = cli.parser().parse_args(["map-flight", "--job", str(self.job)])
        self.assertEqual(args.recognition, "auto")
        with patch("sys.argv", ["fpvsesh", "map-flight", "--job", str(self.job), "--recognition", "thorough"]), \
                patch.object(video_review, "map_flight") as action:
            cli.main()
            action.assert_called_once_with(str(self.job), "thorough")

    def test_partial_model_coverage_has_usable_partial_terminal_status(self):
        self.build.return_value = {"sources": [], "learning": {"video_model": {
            "available": False, "coverage_complete": False, "windows_analyzed": 1,
            "windows_requested": 4, "coverage_seconds": 8,
            "message": "Inference stopped after a decode error; completed observations remain available."}}}
        result = self.run_review()
        self.assertTrue(result["partial"])
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(self.events[-1]["stage"], "partial")
        self.assertEqual(self.events[-1]["windows_analyzed"], 1)
        self.assertEqual(self.events[-1]["windows_requested"], 4)
        self.assertIn("1 of 4", self.events[-1]["message"])
        status = json.loads((self.job / "recognition-status.json").read_text())
        self.assertEqual(status["stage"], "partial")
        self.assertFalse(status["coverage_complete"])
        for name in ("timeline.json", "status.json", "final_4k.mp4", "exports.json", "artifact-state.json"):
            self.assertEqual((self.job / name).read_bytes(), self.before[name])

    def test_unavailable_video_is_partial_but_explicit_off_is_complete(self):
        self.build.return_value = {"sources": [], "learning": {"video_model": {
            "available": False, "windows_analyzed": 0, "message": "Optional video model is not installed"}}}
        self.assertTrue(self.run_review()["partial"])
        self.assertEqual(self.events[-1]["stage"], "partial")
        self.assertIn("unavailable", self.events[-1]["message"])
        self.assertFalse(self.run_review("off")["partial"])
        self.assertEqual(self.events[-1]["stage"], "complete")
        self.assertIn("switched off", self.events[-1]["message"])


if __name__ == "__main__":
    unittest.main()
