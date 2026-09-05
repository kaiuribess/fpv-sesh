"""Flight provenance, learning gates, optional scene fallback and real inference."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fpvsesh import flightmap
from fpvsesh.control import Cancelled
from fpvsesh.media import locate_tools, run


def analysis(name="flight-a", duration=9, identity=None):
    rows = []
    for index in range(int(duration * 6)):
        t = index/6
        rows.append({"t": t, "motion": 12 if t < 6 else .1,
                     "rotation": 35 if t < 3 else 1, "dx": 5, "dy": 2,
                     "proximity": .9, "parallax_confidence": .8})
    return {"source": name, "identity": identity or hashlib.sha256(name.encode()).hexdigest(),
            "duration": duration, "rows": rows}


def unavailable():
    return {"available": False, "message": "Optional scene model missing; motion fallback", "sampled_frames": 0}


class FlightMapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = Path(self.temp.name)
        self.offline = patch.object(flightmap, "scene_model_status", side_effect=unavailable)
        self.offline.start()

    def tearDown(self):
        self.offline.stop()
        self.temp.cleanup()

    def build(self, analyses=None, labels=None):
        return flightmap.build_flight_map(analyses or [analysis()], labels or [], self.cache)

    def test_fresh_clone_map_has_motion_estimates_without_fake_tricks(self):
        result = self.build()
        events = result["sources"][0]["events"]
        self.assertEqual([row["label"] for row in events],
                         ["rotation burst estimate", "close-pass / weave estimate", "low-motion interval"])
        self.assertTrue(all(row["method"] == "motion heuristic" for row in events))
        self.assertFalse(result["learning"]["ready"])
        self.assertFalse(result["learning"]["online_model"]["available"])
        self.assertNotIn("crash", json.dumps(events).lower())
        self.assertNotIn("powerloop", json.dumps(events).lower())
        self.assertTrue((self.cache / "learning/latest-flight-map.json").is_file())

    def test_fractional_tail_stays_in_source_and_scene_is_separate(self):
        a = analysis(duration=7.25)
        a["scene_samples"] = [{"t": 1, "groups": {"woodland": .8, "sky": .1}}]
        with patch.object(flightmap, "_scene_samples", return_value=unavailable()):
            result = self.build([a])
        events = result["sources"][0]["events"]
        self.assertEqual(events[-1]["end"], 7.25)
        self.assertEqual(events[0]["scene"]["label"], "woodland")
        self.assertEqual(events[0]["method"], "motion heuristic")
        self.assertEqual(events[0]["confidence"], .65)

    def test_confirmed_exact_label_is_distinct_and_deduplicated(self):
        a = analysis()
        label = {"source": a["source"], "source_identity": a["identity"], "start": 0, "end": 3, "label": "power loop", "confidence": 1}
        first = self.build([a], [label])
        again = self.build([a], [label])
        self.assertEqual(again["learning"]["examples"], 1)
        self.assertEqual(first["sources"][0]["events"][1]["method"], "user-confirmed")
        candidates = [{"source": a["source"], "start": 0, "end": 3, "score": 42},
                      {"source": a["source"], "start": .2, "end": 2.8, "score": 41}]
        flightmap.annotate_candidates(candidates, [a], self.cache / "learning")
        self.assertEqual(candidates[0]["flight_method"], "user-confirmed")
        self.assertEqual(candidates[0]["flight_label"], "power loop")
        self.assertEqual(candidates[1]["flight_method"], "motion heuristic")
        self.assertEqual([row["score"] for row in candidates], [42, 41])
        stored = json.loads((self.cache / "learning/confirmed-examples.json").read_text())
        self.assertNotIn("source", stored["examples"][0])
        self.assertEqual(stored["examples"][0]["source_identity"], a["identity"])

    def test_learning_requires_three_examples_across_two_source_hashes(self):
        a, b, c = analysis("a"), analysis("b"), analysis("c")
        labels = [{"source": "a", "source_identity": a["identity"], "start": start, "end": end, "label": "confirmed roll", "confidence": 1}
                  for start, end in [(0, 1), (1, 2), (2, 3)]]
        self.assertFalse(self.build([a], labels)["learning"]["ready"])
        # A renamed duplicate recording does not create an independent flight.
        duplicate = analysis("renamed", identity=a["identity"])
        label = {"source": "renamed", "source_identity": a["identity"], "start": .2, "end": 1.2, "label": "confirmed roll", "confidence": 1}
        self.assertFalse(self.build([duplicate], [label])["learning"]["ready"])
        label.update(source="b", source_identity=b["identity"], start=0, end=1)
        result = self.build([a, b], [label])
        self.assertTrue(result["learning"]["ready"])
        self.assertEqual(result["learning"]["enabled_labels"], ["confirmed roll"])
        candidate = {"source": "c", "start": 0, "end": 1}
        flightmap.annotate_candidates([candidate], [c], self.cache / "learning")
        self.assertEqual(candidate["flight_label"], "confirmed roll")
        self.assertIn("nearest neighbors", candidate["flight_method"])
        self.assertLess(candidate["flight_confidence"], 1)

    def test_holdout_removes_all_examples_from_held_out_flight(self):
        examples = [{"source_identity": source, "label": "turn", "features": [value]*8}
                    for source, value in [("a", .1), ("a", .12), ("b", .11)]]
        original = flightmap._predict
        with patch.object(flightmap, "_predict", wraps=original) as spy:
            result = flightmap._learning_status(examples, unavailable())
        training_sources = [[row["source_identity"] for row in call.args[1]] for call in spy.call_args_list]
        self.assertEqual(training_sources, [["b"], ["b"], ["a", "a"]])
        self.assertEqual(result["validation"]["examples"], 3)

    def test_distant_or_conflicting_neighbors_abstain(self):
        examples = [{"source_identity": str(i), "label": name, "features": [value]*8}
                    for i, (name, value) in enumerate([("a", .1), ("b", .1), ("b", .1)])]
        self.assertIsNone(flightmap._predict([3]*8, examples, ["a", "b"]))
        self.assertIsNone(flightmap._predict([.1]*8, examples, ["a", "b"]))

    def test_bad_labels_do_not_replace_existing_dataset(self):
        self.build()
        path = self.cache / "learning/confirmed-examples.json"
        before = path.read_bytes()
        base = {"source": "flight-a", "source_identity": analysis()["identity"], "start": 0, "end": 1, "label": "roll", "confidence": 1}
        for update in [{"source": "missing"}, {"start": float("nan")}, {"end": 10},
                       {"confidence": .7}, {"label": ""}, {"start": -1}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.build(labels=[{**base, **update}])
            self.assertEqual(path.read_bytes(), before)

    def test_initial_cancel_preserves_previous_map(self):
        self.build()
        path = self.cache / "learning/latest-flight-map.json"
        before = path.read_bytes()
        with self.assertRaises(Cancelled):
            flightmap.build_flight_map([analysis()], [], self.cache,
                                      checkpoint=lambda: (_ for _ in ()).throw(Cancelled()))
        self.assertEqual(path.read_bytes(), before)

    def test_replaced_source_or_unbound_confirmation_cannot_teach_new_footage(self):
        original = analysis("same-path.mp4")
        label = {"source": original["source"], "source_identity": original["identity"],
                 "start": 0, "end": 3, "label": "confirmed roll", "confidence": 1}
        self.build([original], [label])
        path = self.cache / "learning/confirmed-examples.json"
        before = path.read_bytes()
        replacement = {**original, "identity": "a"*64}
        result = self.build([replacement], [label])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(result["learning"]["examples"], 1)
        self.assertIn("stale label", result["learning"]["message"])
        self.assertTrue(all(row["method"] != "user-confirmed" for row in result["sources"][0]["events"]))
        label.pop("source_identity")
        result = self.build([original], [label])
        self.assertTrue(result["learning"]["warnings"])
        self.assertEqual(path.read_bytes(), before)

    def test_cache_rejects_incomplete_nonfinite_or_stale_samples(self):
        signature = {"source": "a", "model": "b"}
        sample = {"t": 0, "scene": "sky", "score": .8, "groups": {"sky": .8},
                  "top_classes": [{"label": "sky", "score": .8}]}
        record = {"signature": signature, "samples": [sample, {**sample, "t": 1}],
                  "coverage_seconds": 1.5, "proxy_fps": 12}
        self.assertTrue(flightmap._valid_samples(record, signature, 1.5))
        self.assertFalse(flightmap._valid_samples(record, {**signature, "source": "changed"}, 1.5))
        for broken in [{**record, "samples": [sample]},
                       {**record, "samples": [sample, {**sample, "t": float("nan")}]},
                       {**record, "samples": [sample, {**sample, "t": 1, "groups": {"sky": float("nan")}}]}]:
            self.assertFalse(flightmap._valid_samples(broken, signature, 1.5))

    def test_fractional_proxy_rounding_accepts_full_decoded_coverage(self):
        sample = {"scene": "sky", "score": .8, "groups": {"sky": .8},
                  "top_classes": [{"label": "sky", "score": .8}]}
        record = {"signature": {"source": "a"}, "samples": [{"t": i, **sample} for i in range(6)],
                  "coverage_seconds": 6.0, "proxy_fps": 12}
        self.assertTrue(flightmap._valid_samples(record, record["signature"], 6.016667))
        self.assertFalse(flightmap._valid_samples(record, record["signature"], 6.5))
        self.assertFalse(flightmap._valid_samples({**record, "samples": record["samples"][:-1]}, record["signature"], 6.016667))

    def test_unavailable_or_changed_model_clears_previously_attached_scene(self):
        a = analysis()
        a["scene_samples"] = [{"t": 0, "scene": "old scene"}]
        self.build([a])
        self.assertNotIn("scene_samples", a)

    def test_corrupt_local_example_cache_is_ignored(self):
        folder = self.cache / "learning"
        folder.mkdir()
        path = folder / "confirmed-examples.json"
        path.write_text(json.dumps({"version": 1, "examples": [{"features": ["bad"]*8}, {"features": [0]*8}]}))
        candidate = {"source": "flight-a", "start": 0, "end": 3}
        flightmap.annotate_candidates([candidate], [analysis()], folder)
        self.assertEqual(candidate["flight_method"], "motion heuristic")
        result = self.build()
        self.assertEqual(result["learning"]["examples"], 0)
        self.assertIn("malformed", result["learning"]["message"])

    def test_cancel_terminates_running_worker_and_removes_control_files(self):
        proxy = self.cache / "proxy.mp4"
        proxy.write_bytes(b"owned synthetic placeholder")
        a = {**analysis(duration=2), "proxy": str(proxy)}
        process = MagicMock()
        process.poll.return_value = None
        calls = 0
        def checkpoint():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise Cancelled("test cancellation")
        with patch.object(flightmap, "scene_model_status", return_value={"available": True, "model_hash": "test", "message": "ready"}), \
             patch.object(flightmap.subprocess, "Popen", return_value=process), self.assertRaises(Cancelled):
            flightmap._scene_samples([a], self.cache, lambda *args: None, checkpoint)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()
        self.assertFalse(list((self.cache / "learning/scenes").glob("worker-*.json")))
        self.assertEqual(proxy.read_bytes(), b"owned synthetic placeholder")


class OptionalRealSceneTests(unittest.TestCase):
    @unittest.skipUnless(flightmap.scene_model_status()["available"], "Optional verified scene model is not installed")
    def test_real_safe_model_inference_unicode_source_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            source = cache / "树🌲' proxy.mp4"
            ffmpeg, _ = locate_tools()
            run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=12:duration=1.25",
                 "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            a = {**analysis(str(source), 1.25, digest), "proxy": str(source)}
            status = flightmap._scene_samples([a], cache, lambda *args: None, lambda: None)
            self.assertTrue(status["available"], status["message"])
            self.assertEqual(status["sampled_frames"], 2)
            self.assertEqual([row["t"] for row in a["scene_samples"]], [0, 1])
            self.assertTrue(all(len(row["top_classes"]) == 5 for row in a["scene_samples"]))
            with patch.object(flightmap.subprocess, "Popen") as process:
                reused = flightmap._scene_samples([a], cache, lambda *args: None, lambda: None)
            process.assert_not_called()
            self.assertEqual(reused["sampled_frames"], 2)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            # Changed source identity must not inherit a previous inference record.
            a["identity"] = "different-source"
            with patch.object(flightmap.subprocess, "Popen", side_effect=OSError("intentional unavailable runtime")) as process:
                changed = flightmap._scene_samples([a], cache, lambda *args: None, lambda: None)
            process.assert_called_once()
            self.assertFalse(changed["available"])
            self.assertNotIn("scene_samples", a)


if __name__ == "__main__":
    unittest.main()
