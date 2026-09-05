"""UI contract checks using isolated jobs; no media rendering or external playback."""
import json
import gc
import os
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fpvsesh.ui import SeshApp, SOCIAL_FORMATS, write_json, read_json


class UiSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The application has one Tk interpreter per process. Exercise that
        # lifetime with fresh jobs/state, instead of repeatedly unloading Tk.
        cls.ui_directory = tempfile.TemporaryDirectory(prefix="fpvsesh-ui-window-")
        cls.thumbnail_stub = patch.object(SeshApp, "_request_thumbnail", return_value=None)
        cls.thumbnail_stub.start()
        with patch.object(SeshApp, "_seed_inputs", lambda app: app._refresh_inputs()):
            cls.shared_app = SeshApp(Path(cls.ui_directory.name))
        cls.shared_app.attributes("-alpha", 0)
        cls.shared_app.after_cancel(cls.shared_app._poll_after)
        cls.shared_app._poll_after = None

    @classmethod
    def tearDownClass(cls):
        cls.shared_app.destroy()
        cls.shared_app = None
        cls.thumbnail_stub.stop()
        gc.collect()
        cls.ui_directory.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fpvsesh-ui-test-")
        self.root = Path(self.directory.name)
        self.app = self.shared_app
        self.app.app_dir = self.root
        self.app.process = None
        self.app.events = queue.Queue()
        self.app.files, self.app.folder = [], None
        self.app.candidates = []
        self.app.override_choices = {"keep": [], "exclude": []}
        self.app._candidate_mtime = self.app._flight_mtime = self.app._last_diagnostics = None
        self.app._flight_data, self.app._flight_rows = {}, []
        self.app.mapping_only = self.app.run_preview_only = self.app.paused = False
        self.app.terminal_stage = ""
        self.app.last_args = None
        self.app.warnings_seen.clear()
        self.app.geometry("1020x650")
        self.job = self.root / "output" / "test-job"
        self.job.mkdir(parents=True)
        self.source = self.root / "recording.mp4"
        self.source.touch()
        self.app.job_dir = self.job
        self.identity = "a" * 64
        write_json(self.job / "sources.json", [{"source": str(self.source), "duration": 30, "sha256": self.identity}])
        self.app._refresh_inputs()
        self.app._restore_settings(self.job)
        self.app._paint_candidates()
        self.app._load_flight_map(force=True)
        self.app.flight_filter_value.set("All motion")
        self.app._set_busy(False)
        self.app.notebook.select(self.app._pages["Session"])
        self.app._poll_after = self.app.after(120, self.app._poll)
        self.app.update()

    def tearDown(self):
        if self.app._poll_after:
            self.app.after_cancel(self.app._poll_after)
            self.app._poll_after = None
        self.app.process = None
        self.directory.cleanup()

    @staticmethod
    def value(args, flag):
        return args[args.index(flag) + 1]

    def test_defaults_and_explicit_music_removal(self):
        args = self.app._command_args()
        self.assertIn("--no-music", args)
        self.assertNotIn("--music", args)
        self.assertEqual(self.value(args, "--look"), "natural")
        self.assertEqual(float(self.value(args, "--strength")), 0)
        self.assertEqual(self.value(args, "--social-formats"), "none")
        self.assertEqual(self.value(args, "--recognition"), "auto")
        self.assertIn("--beat-sync", args)
        track = self.root / "music.wav"
        track.touch()
        self.app.music_path = track
        self.assertEqual(self.value(self.app._settings_args(), "--music"), str(track))
        self.app._remove_music()
        self.assertTrue(self.app.settings_dirty)
        self.assertIn("--no-music", self.app._settings_args())

    def test_all_settings_restore_and_regenerate(self):
        track = self.root / "music.mp3"
        track.touch()
        write_json(self.job / "settings.json", {
            "inputs": [str(self.source)], "music": str(track), "music_level": .61, "music_offset": 13.25,
            "music_fade": 2.25, "music_end": "loop", "beat_sync": False, "social_formats": list(SOCIAL_FORMATS),
            "framing": "fill", "focus_x": .7, "edit_order": "chronological", "recovery": 4.5,
            "duration": "180", "style": "flow", "look": "natural", "strength": 0, "quality": "lanczos", "codec": "h264",
            "recognition": "thorough",
        })
        self.app._restore_settings(self.job)
        self.assertFalse(self.app.settings_dirty)
        with patch.object(self.app, "_launch") as launch:
            self.app._regenerate()
        args = launch.call_args.args[0]
        for flag, expected in (("--music", str(track)), ("--music-end", "loop"), ("--social-formats", "vertical,square,portrait"),
                               ("--framing", "fill"), ("--edit-order", "chronological"), ("--recovery", "4.5"),
                               ("--duration", "180"), ("--style", "flow"), ("--codec", "h264"), ("--recognition", "thorough")):
            self.assertEqual(self.value(args, flag), expected)
        self.assertIn("--no-beat-sync", args)
        self.assertNotIn("--input", args)
        self.assertNotIn("--no-music", args)
        self.assertEqual(float(self.value(args, "--focus-x")), .7)
        write_json(self.job / "settings.json", {"inputs": [str(self.source)]})
        self.app._restore_settings(self.job)
        self.assertIsNone(self.app.music_path)
        self.assertEqual(self.value(self.app._settings_args(), "--social-formats"), "none")
        self.assertEqual(self.value(self.app._settings_args(), "--recognition"), "auto")

    def test_recognition_settings_roundtrip_and_mark_job_dirty(self):
        for value, label in (("auto", "Automatic"), ("off", "Off"), ("thorough", "Thorough")):
            write_json(self.job / "settings.json", {"recognition": value})
            self.app._restore_settings(self.job)
            self.assertEqual(self.app.recognition_value.get(), label)
            self.assertEqual(self.value(self.app._settings_args(), "--recognition"), value)
            self.assertFalse(self.app.settings_dirty)
            self.assertFalse(self.app.recognition_dirty)
        self.app.recognition_value.set("Off")
        self.assertFalse(self.app.settings_dirty)
        self.assertTrue(self.app.recognition_dirty)
        self.assertIn("Runs locally using an internet-trained model", self.app.recognition_note.cget("text"))

    def test_refresh_understanding_launches_only_map_command_without_confirmation(self):
        self.app.recognition_value.set("Thorough")
        with patch.object(self.app, "_launch") as launch, patch("fpvsesh.ui.messagebox.askyesno") as confirmation:
            self.app._refresh_understanding()
        launch.assert_called_once_with(["map-flight", "--job", str(self.job), "--recognition", "thorough"])
        confirmation.assert_not_called()
        self.assertNotIn("make", launch.call_args.args[0])
        self.assertNotIn("--input", launch.call_args.args[0])

    def test_refresh_understanding_requires_job_and_idle_process(self):
        self.app.job_dir = None
        self.app._refresh_outputs()
        self.assertTrue(self.app.map_button.instate(["disabled"]))
        with patch.object(self.app, "_launch") as launch:
            self.app._refresh_understanding()
        launch.assert_not_called()
        self.app.job_dir = self.job
        self.app._refresh_outputs()
        self.assertTrue(self.app.map_button.instate(["!disabled"]))
        self.app.process = MagicMock()
        self.app._refresh_outputs()
        self.assertTrue(self.app.map_button.instate(["disabled"]))
        with patch.object(self.app, "_launch") as launch:
            self.app._refresh_understanding()
        launch.assert_not_called()
        self.app.process = None

    def test_recognition_only_change_keeps_finished_exports_playable(self):
        (self.job / "preview.mp4").touch()
        (self.job / "final_4k.mp4").touch()
        write_json(self.job / "status.json", {"stage": "complete"})
        self.app._refresh_outputs()
        self.app.recognition_value.set("Thorough")
        self.assertTrue(self.app.recognition_dirty)
        self.assertFalse(self.app.settings_dirty)
        self.assertTrue(self.app.play_final_button.instate(["!disabled"]))
        self.app.look_value.set("Cinematic")
        self.app.recognition_value.set("Off")
        self.assertTrue(self.app.settings_dirty)
        self.assertTrue(self.app.play_final_button.instate(["disabled"]))

    def test_map_process_uses_shared_pause_cancel_controls(self):
        process = MagicMock()
        with patch("fpvsesh.ui.subprocess.Popen", return_value=process) as opening, patch("fpvsesh.ui.threading.Thread"):
            self.app._refresh_understanding()
        self.assertTrue(self.app.mapping_only)
        self.assertIn("map-flight", opening.call_args.args[0])
        self.assertNotIn("make", opening.call_args.args[0])
        self.assertIn("Refreshing flight", self.app.stage_text.get())
        self.assertTrue(self.app.map_button.instate(["disabled"]))
        self.assertTrue(self.app.pause_button.instate(["!disabled"]))
        self.app._pause()
        self.assertEqual(read_json(self.root / "cache/control.json"), {"action": "pause"})
        self.app._pause()
        self.assertEqual(read_json(self.root / "cache/control.json"), {"action": "resume"})
        self.app._cancel()
        self.assertEqual(read_json(self.root / "cache/control.json"), {"action": "cancel"})
        self.app.process = None

    def test_map_completion_refreshes_labels_without_mutating_finished_edit_or_pending_choices(self):
        preserved = {}
        for name in ("timeline.json", "exports.json", "artifact-state.json"):
            write_json(self.job / name, {"unchanged": name})
            preserved[name] = (self.job / name).read_bytes()
        write_json(self.job / "status.json", {"stage": "complete"})
        preserved["status.json"] = (self.job / "status.json").read_bytes()
        (self.job / "final_4k.mp4").write_bytes(b"finished video fixture")
        preserved["final_4k.mp4"] = (self.job / "final_4k.mp4").read_bytes()
        original = {"id": "same-cut", "source": str(self.source), "start": 3, "end": 9, "selected": True, "score": 80}
        self.app.candidates = [original]
        self.app._paint_candidates()
        self.app.table.selection_set("0")
        self.app.override_choices = {"keep": [], "exclude": ["same-cut"]}
        self.app.overrides_dirty = True
        write_json(self.job / "ui-overrides.json", self.app.override_choices)
        os.utime(self.job / "ui-overrides.json", ns=(1_000_000_000, 1_000_000_000))
        write_json(self.job / "candidates.json", {"candidates": [{**original, "trick_label": "flip", "trick_status": "suggested",
                                                                  "trick_evidence": "Horizon movement across frames"}],
                                                  "overrides": {"keep": [], "exclude": []}})
        write_json(self.job / "flight-map.json", {"sources": [], "learning": {"video_model": {"mode": "auto", "available": True}}})
        self.app.mapping_only = True
        self.app.recognition_dirty = True
        self.app.settings_dirty = True
        self.app.process = MagicMock()
        self.app.terminal_stage = "complete"
        self.app.after_cancel(self.app._poll_after)
        self.app._poll_after = None
        self.app.events.put(("exit", 0))
        with patch.object(self.app, "_load_job_poster") as poster:
            self.app._poll()
        poster.assert_not_called()
        self.assertIn("understanding updated", self.app.stage_text.get())
        self.assertIn("edit is unchanged", self.app.stage_text.get())
        self.assertFalse(self.app.recognition_dirty)
        self.assertTrue(self.app.settings_dirty)
        self.assertTrue(self.app.overrides_dirty)
        self.assertEqual(self.app.override_choices["exclude"], ["same-cut"])
        self.assertEqual(self.app.table.selection(), ("0",))
        self.assertIn("flip · Estimate", self.app.table.item("0", "values")[-1])
        for name, original_bytes in preserved.items():
            self.assertEqual((self.job / name).read_bytes(), original_bytes)

    def test_map_cancel_does_not_claim_finished_render_or_clear_pending_recognition(self):
        self.app.mapping_only = True
        self.app.recognition_dirty = True
        self.app.process = MagicMock()
        self.app.terminal_stage = "cancelled"
        self.app.after_cancel(self.app._poll_after)
        self.app._poll_after = None
        self.app.events.put(("exit", 0))
        self.app._poll()
        self.assertIn("analysis cancelled", self.app.stage_text.get())
        self.assertTrue(self.app.recognition_dirty)
        self.assertTrue(self.app.map_button.instate(["!disabled"]))

    def test_map_partial_completion_retains_usable_results_without_claiming_complete_coverage(self):
        self.app.mapping_only = True
        self.app.recognition_dirty = True
        self.app.process = MagicMock()
        self.app.after_cancel(self.app._poll_after)
        self.app._poll_after = None
        self.app.events.put(("event", {"stage": "partial", "progress": 1, "operation": "map-flight",
                                        "message": "5 of 83 windows reviewed; completed observations are available."}))
        self.app.events.put(("exit", 0))
        self.app._poll()
        self.assertEqual(self.app.terminal_stage, "partial")
        self.assertIn("partly updated", self.app.stage_text.get())
        self.assertIn("observations are available", self.app.stage_text.get())
        self.assertIn("finished edit is unchanged", self.app.detail_text.get())
        self.assertNotIn("understanding updated.", self.app.stage_text.get())
        self.assertFalse(self.app.recognition_dirty)
        self.assertTrue(self.app.map_button.instate(["!disabled"]))

    def test_invalid_numbers_and_missing_music_are_rejected(self):
        for invalid in ("nan", "inf", "-1", "abc"):
            self.app.music_offset_value.set(invalid)
            with self.assertRaises(ValueError):
                self.app._settings_args()
        self.app.music_offset_value.set("0")
        self.app.music_path = self.root / "missing.mp3"
        with self.assertRaisesRegex(ValueError, "missing"):
            self.app._settings_args()

    def test_current_output_gates_and_social_paths(self):
        (self.job / "preview.mp4").touch()
        (self.job / "final_4k.mp4").touch()
        (self.job / "social").mkdir()
        (self.job / "social" / "vertical.mp4").touch()
        write_json(self.job / "status.json", {"stage": "complete"})
        self.app._refresh_outputs()
        self.assertTrue(self.app.play_final_button.instate(["!disabled"]))
        self.assertEqual(list(self.app._playback_paths.values()), [self.job / "social" / "vertical.mp4"])
        with patch.object(self.app, "_open_path") as opening:
            self.app._play_social()
            opening.assert_called_once_with(self.job / "social" / "vertical.mp4")
        self.app.music_fade_value.set("2")
        self.assertTrue(self.app.settings_dirty)
        self.assertTrue(self.app.social_play_button.instate(["disabled"]))
        self.assertTrue(self.app.play_final_button.instate(["disabled"]))
        with patch.object(self.app, "_open_path") as opening:
            self.app._play_social()
            self.app._play_final()
            opening.assert_not_called()

    def test_exact_ranges_and_teaching_do_not_force_ground_into_edit(self):
        with self.assertRaises(ValueError):
            self.app._save_exact_range(str(self.source), 20, 31)
        self.app._save_exact_range(str(self.source), 8.5, 15)
        self.assertTrue(read_json(self.job / "reviewed-intervals.json")[0]["keep"])
        self.assertEqual(read_json(self.job / "reviewed-intervals.json")[0]["source_identity"], self.identity)
        self.app.candidates = [{"id": "ground", "source": str(self.source), "identity": self.identity, "start": 25, "end": 29, "score": 1}]
        self.app._paint_candidates()
        self.app.table.selection_set("0")
        self.app.flight_label_value.set("landing")
        self.app._teach_moment()
        labels = read_json(self.job / "flight-labels.json")
        self.assertEqual(labels, [{"source": str(self.source), "source_identity": self.identity, "start": 25, "end": 29, "label": "landing", "confidence": 1.0}])
        reviews = read_json(self.job / "reviewed-intervals.json")
        self.assertFalse(reviews[-1]["keep"])
        self.assertEqual(self.app.override_choices["keep"], [])
        self.assertTrue(self.app.settings_dirty)
        self.assertEqual(self.app._parse_time("1:02.5"), 62.5)
        with self.assertRaises(ValueError):
            self.app._parse_time("1:70")

    def test_unbound_or_replaced_recording_cannot_teach(self):
        self.app.candidates = [{"id": "unknown", "source": str(self.root / "other.mp4"), "start": 1, "end": 4}]
        self.app._paint_candidates()
        self.app.table.selection_set("0")
        with patch("fpvsesh.ui.messagebox.showinfo") as notice:
            self.app._teach_moment()
        notice.assert_called_once()
        self.assertFalse((self.job / "flight-labels.json").exists())
        write_json(self.job / "sources.json", [{"source": str(self.source), "duration": 30, "sha256": self.identity, "size_bytes": 7}])
        with self.assertRaisesRegex(ValueError, "changed"):
            self.app._save_exact_range(str(self.source), 1, 4)

    def test_old_candidate_cannot_bind_to_new_probe_identity(self):
        write_json(self.job / "sources.json", [{"source": str(self.source), "duration": 30, "sha256": "b" * 64,
                                              "size_bytes": self.source.stat().st_size, "mtime_ns": self.source.stat().st_mtime_ns}])
        self.app.candidates = [{"id": "stale", "source": str(self.source), "identity": self.identity, "start": 1, "end": 4}]
        self.app._paint_candidates()
        self.app.table.selection_set("0")
        with patch("fpvsesh.ui.messagebox.showinfo") as notice:
            self.app._teach_moment()
        self.assertIn("earlier version", notice.call_args.args[1])
        self.assertFalse((self.job / "flight-labels.json").exists())
        self.assertFalse((self.job / "reviewed-intervals.json").exists())

    def test_exclude_and_clear_release_forced_exact_range(self):
        record = self.app._save_exact_range(str(self.source), 8, 15)
        self.app.candidates = [{"id": "exact", "review_key": record["key"], "source": str(self.source), "start": 8, "end": 15}]
        self.app._paint_candidates()
        self.app.table.selection_set("0")
        self.app._set_choice("exclude")
        self.assertFalse(read_json(self.job / "reviewed-intervals.json")[0]["keep"])
        self.assertEqual(read_json(self.job / "ui-overrides.json")["exclude"], ["exact"])
        self.app.table.selection_set("0")
        self.app._set_choice("keep")
        self.assertTrue(read_json(self.job / "reviewed-intervals.json")[0]["keep"])
        self.app._reset_choices()
        self.assertFalse(read_json(self.job / "reviewed-intervals.json")[0]["keep"])
        self.assertEqual(read_json(self.job / "ui-overrides.json"), {"keep": [], "exclude": []})

    def test_rendered_override_state_replaces_old_pending_file(self):
        write_json(self.job / "ui-overrides.json", {"keep": [], "exclude": []})
        write_json(self.job / "candidates.json", {"candidates": [], "overrides": {"keep": ["manual-range"], "exclude": []}})
        os.utime(self.job / "ui-overrides.json", ns=(1_000_000_000, 1_000_000_000))
        self.app._load_candidates(force=True)
        self.assertEqual(self.app.override_choices["keep"], ["manual-range"])
        self.assertFalse(self.app.overrides_dirty)

    def test_final_render_preserves_canonical_review_order(self):
        canonical = {"keep": ["b", "a"], "exclude": [], "order": ["b", "a"]}
        write_json(self.job / "overrides.json", canonical)
        self.app.settings_dirty = self.app.overrides_dirty = False
        with patch.object(self.app, "_launch") as launch:
            self.app._render_final()
        args = launch.call_args.args[0]
        self.assertNotIn("--overrides", args)
        self.assertIn("--no-music", args)
        self.assertEqual(read_json(self.job / "overrides.json"), canonical)
        self.assertFalse((self.job / "ui-overrides.json").exists())

    def test_flight_map_shows_evidence(self):
        write_json(self.job / "flight-map.json", {"sources": [{"source": str(self.source), "duration": 30, "events": [
            {"start": 5, "end": 8, "label": "tree weave", "confidence": .6, "method": "local examples", "reason": "estimated match"}]}],
            "learning": {"examples": 3, "ready": False, "message": "More examples needed."}})
        self.app._load_flight_map(force=True)
        self.assertEqual(len(self.app.flight_table.get_children()), 1)
        self.assertIn("More examples needed", self.app.learning_text.get())
        values = self.app.flight_table.item("0", "values")
        self.assertEqual(values[4], "Estimate")
        self.assertNotIn("%", " ".join(values))
        self.assertEqual(self.app.flight_table.heading("confidence", "text"), "Evidence")

    def test_video_recognition_filters_and_details_retain_correct_source_mapping(self):
        model = "Qwen3-VL-2B-Instruct"
        video_events = [{"start": start, "end": end, "label": label, "raw_label": label,
                         "status": status, "evidence": evidence, "model": model,
                         "method": "online-pretrained video model", "checks": ["Motion evidence considered"]}
                        for start, end, label, status, evidence in (
                            (0, 8, "ordinary flight", "suggested", "Steady forward movement"),
                            (8, 16, "flip", "suggested", "Horizon rotates through the sequence"),
                            (16, 24, "roll", "uncertain", "Insufficient follow-through; https://example.invalid is plain model text"),
                            (24, 30, "tree weaving", "suggested", "Passes around young trees"))]
        write_json(self.job / "flight-map.json", {
            "sources": [{"source": str(self.source), "duration": 30,
                         "events": [{"start": 2, "end": 5, "label": "rotation burst estimate", "confidence": .65, "method": "motion heuristic"}],
                         "video_events": video_events}],
            "learning": {"examples": 0, "video_model": {"available": True, "name": model,
                "message": "Review suggested labels.", "windows_analyzed": 4, "coverage_seconds": 30, "mode": "auto"}}})
        self.app._load_flight_map(force=True)
        self.assertEqual(len(self.app.flight_table.get_children()), 5)
        self.assertIn("Online-trained video", self.app.learning_text.get())
        self.assertIn("4 windows reviewed", self.app.learning_text.get())
        self.assertIn("30s covered", self.app.learning_text.get())
        self.assertIn("0 optional local examples", self.app.learning_text.get())
        self.app.flight_filter_value.set("Possible tricks")
        self.assertEqual(len(self.app.flight_table.get_children()), 1)
        self.assertEqual(self.app.flight_table.item("0", "values")[3:5], ("flip", "Estimate"))
        self.app._select_flight_event(0)
        with patch("fpvsesh.ui.messagebox.showinfo") as notice:
            self.app._show_flight_event()
        self.assertIn("00:08.00–00:16.00", notice.call_args.args[1])
        self.assertIn(model, notice.call_args.args[1])
        self.app.flight_filter_value.set("Ordinary flight")
        self.assertEqual([self.app.flight_table.item(row, "values")[3] for row in self.app.flight_table.get_children()],
                         ["ordinary flight", "tree weaving"])
        self.app.flight_filter_value.set("Uncertain")
        self.assertEqual(self.app.flight_table.item("0", "values")[3:5], ("roll", "Uncertain"))
        self.app._select_flight_event(0)
        with patch("fpvsesh.ui.messagebox.showinfo") as notice, patch.object(self.app, "_open_path") as opening:
            self.app._show_flight_event()
        self.assertIn("https://example.invalid", notice.call_args.args[1])
        opening.assert_not_called()

    def test_model_estimates_never_display_as_confirmed_or_percentage_accuracy(self):
        self.app.candidates = [{"id": "trick", "source": str(self.source), "start": 3, "end": 9, "score": 80,
                                "trick_label": "powerloop", "trick_status": "suggested", "confidence": 1,
                                "trick_evidence": "Rising arc and inverted horizon", "reason": "Recovery retained"},
                               {"id": "confirmed", "source": str(self.source), "start": 12, "end": 18,
                                "flight_label": "roll", "flight_method": "user-confirmed", "confidence": 1}]
        self.app._paint_candidates()
        inferred = self.app.table.item("0", "values")[-1]
        confirmed = self.app.table.item("1", "values")[-1]
        self.assertTrue(inferred.startswith("powerloop · Estimate"))
        self.assertIn("Rising arc", inferred)
        self.assertIn("Recovery retained", inferred)
        self.assertNotIn("%", inferred)
        self.assertIn("roll · Confirmed", confirmed)
        self.assertIn("User confirmation", confirmed)

    def test_measured_rotation_suggestion_keeps_model_miss_and_method_distinct(self):
        method = "measured image rotation with online-pretrained video context"
        checks = ["The video model did not independently identify the roll; movement evidence supplies this suggestion"]
        ordinary = {"start": 0, "end": 8, "label": "ordinary flight", "raw_label": "ordinary flight",
                    "status": "suggested", "method": "online-pretrained video model", "model": "Qwen3-VL-2B-Instruct",
                    "evidence": "The model described forward flight.", "checks": []}
        measured = {**ordinary, "label": "roll", "method": method,
                    "evidence": "Feature tracking supports a possible roll, not verified drone attitude.", "checks": checks}
        write_json(self.job / "flight-map.json", {"sources": [{"source": str(self.source), "duration": 30,
                                                                "video_events": [ordinary, measured]}]})
        self.app._load_flight_map(force=True)
        self.assertEqual(len(self.app.flight_table.get_children()), 2)
        self.app.flight_filter_value.set("Possible tricks")
        self.assertEqual(len(self.app.flight_table.get_children()), 1)
        values = self.app.flight_table.item("0", "values")
        self.assertEqual(values[3:5], ("roll", "Estimate"))
        self.assertTrue(values[5].startswith(method))
        self.assertIn("Video model originally reported: ordinary flight", values[5])
        self.assertIn(checks[0], values[5])
        self.assertNotIn("Online-trained video ·", values[5])
        self.app.flight_filter_value.set("Ordinary flight")
        self.assertEqual(self.app.flight_table.item("0", "values")[3], "ordinary flight")
        candidate = {"trick_label": "roll", "trick_status": "suggested", "trick_method": method,
                     "trick_model": ordinary["model"], "trick_raw_label": "ordinary flight", "trick_checks": checks,
                     "trick_evidence": measured["evidence"], "flight_label": "rotation burst estimate", "flight_method": "motion heuristic"}
        label, status, description = self.app._event_summary(candidate)
        self.assertEqual((label, status), ("roll", "Estimate"))
        self.assertTrue(description.startswith(method))
        self.assertIn("originally reported: ordinary flight", description)
        self.assertIn(checks[0], description)

    def test_watch_section_uses_filtered_source_and_raw_bounds_with_context(self):
        second_source = self.root / "第二 flight.mp4"
        second_source.touch()
        data = {"sources": [
            {"source": str(self.source), "duration": 30, "video_events": [
                {"start": 0, "end": 8, "label": "ordinary flight", "status": "suggested"}]},
            {"source": str(second_source), "identity": "b" * 64, "duration": 30, "video_events": [
                {"start": 19.5, "end": 25.25, "label": "flip", "status": "suggested", "method": "online-pretrained video model"}]}]}
        write_json(self.job / "flight-map.json", data)
        self.app._load_flight_map(force=True)
        self.app.flight_filter_value.set("Possible tricks")
        self.app._select_flight_event(0)
        self.assertTrue(self.app.watch_section_button.instate(["!disabled"]))
        # A refreshed map may insert a different source before the selection;
        # playback must still follow the exact selected source/time, not row 0.
        data["sources"][0]["video_events"].append({"start": 9, "end": 17, "label": "roll", "status": "suggested"})
        write_json(self.job / "flight-map.json", data)
        self.app._load_flight_map(force=True)
        self.assertEqual(self.app.flight_table.selection(), ("1",))
        with patch("fpvsesh.source_review.play_section", return_value={
                "source": str(second_source), "start": 17.5, "end": 27.25}) as playing, \
             patch.object(self.app, "_launch") as rendering:
            self.app._watch_flight_section()
        playing.assert_called_once_with(str(second_source), 19.5, 25.25, source_duration=30, context=2.0, app_dir=self.root)
        rendering.assert_not_called()
        self.assertIn("00:17.50–00:27.25", self.app.detail_text.get())
        self.app.flight_filter_value.set("Uncertain")
        self.assertTrue(self.app.watch_section_button.instate(["disabled"]))

    def test_watch_section_requires_selection_and_reports_missing_player(self):
        with patch("fpvsesh.source_review.play_section") as playing:
            self.app._watch_flight_section()
        playing.assert_not_called()
        write_json(self.job / "flight-map.json", {"sources": [{"source": str(self.source), "duration": 30,
            "events": [{"start": 3, "end": 6, "label": "rotation burst estimate", "method": "motion heuristic"}]}]})
        self.app._load_flight_map(force=True)
        self.app._select_flight_event(0)
        self.app.process = MagicMock()
        self.app.mapping_only = False
        self.app._refresh_watch_section()
        self.assertTrue(self.app.watch_section_button.instate(["disabled"]))
        with patch("fpvsesh.source_review.play_section") as playing:
            self.app._watch_flight_section()
        playing.assert_not_called()
        self.app.mapping_only = True
        self.app._refresh_watch_section()
        self.assertTrue(self.app.watch_section_button.instate(["!disabled"]))
        with patch("fpvsesh.source_review.play_section", side_effect=FileNotFoundError("Bundled player is missing; rerun setup.")), \
             patch("fpvsesh.ui.messagebox.showerror") as notice:
            self.app._watch_flight_section()
        self.assertEqual(notice.call_args.args, ("Could not watch this section", "Bundled player is missing; rerun setup."))

    def test_off_or_missing_video_model_status_is_explicit_and_examples_optional(self):
        for video, expected in (({"available": False, "mode": "off"}, "Video recognition is off"),
                                ({"available": False, "mode": "auto", "message": "Install the video model first."}, "Install the video model first")):
            write_json(self.job / "flight-map.json", {"sources": [], "learning": {"examples": 0, "video_model": video}})
            self.app._load_flight_map(force=True)
            self.assertIn(expected, self.app.learning_text.get())
            self.assertIn("optional local examples", self.app.learning_text.get())

    def test_long_model_diagnostic_keeps_small_window_results_visible_and_logs_details(self):
        diagnostic = ("Video inference could not finish; completed observations remain available. \n"
                      + '  File "recognition_worker.py", line 150, in analyze\n' * 20
                      + "ValueError: Rotation window ended early: expected 150 frames, decoded 149")
        sources = [{"source": f"recording-{index}.mp4", "duration": 200, "events": [
            {"start": 5, "end": 8, "label": "rotation burst estimate", "method": "motion heuristic"}]
        } for index in range(3)]
        self.app.notebook.select(self.app._pages["Flight map"])
        for available in (False, True):
            write_json(self.job / "flight-map.json", {"sources": sources, "learning": {
                "examples": 0, "video_model": {"available": available, "name": "Qwen3-VL-2B-Instruct",
                    "message": diagnostic, "windows_analyzed": 5, "coverage_seconds": 40, "mode": "auto"}}})
            self.app._load_flight_map(force=True)
            self.app.update()
            self.assertIn("See Activity for details", self.app.learning_text.get())
            self.assertNotIn("ValueError", self.app.learning_text.get())
            self.assertNotIn("\n", self.app.learning_text.get())
            self.assertIn(diagnostic, self.app.log_box.get("1.0", "end"))
            self.assertTrue(self.app.map_button.winfo_ismapped())
            self.assertTrue(self.app.flight_filter_combo.winfo_ismapped())
            self.assertTrue(self.app.flight_table.winfo_ismapped())
            self.assertGreaterEqual(self.app.flight_table.winfo_height(), 60)
            self.assertLessEqual(self.app.flight_table.winfo_y() + self.app.flight_table.winfo_height(),
                                 self.app._pages["Flight map"].winfo_height())

    def test_controls_fit_small_window(self):
        for tab in self.app.notebook.tabs():
            self.app.notebook.select(tab)
            self.app.update()
            page = self.app.nametowidget(tab)
            for child in page.winfo_children():
                if child.winfo_ismapped():
                    self.assertLessEqual(child.winfo_y() + child.winfo_height(), page.winfo_height(), self.app.notebook.tab(tab, "text"))
                    self.assertLessEqual(child.winfo_x() + child.winfo_width(), page.winfo_width(), self.app.notebook.tab(tab, "text"))
        for button in (self.app.make_button, self.app.preview_button, self.app.play_final_button):
            self.assertTrue(button.winfo_ismapped())
            self.assertGreaterEqual(button.winfo_width(), button.winfo_reqwidth())

    def test_social_playback_is_reachable_by_scrolling_and_nav_updates(self):
        page = self.app._pages["Social exports"]
        self.app.notebook.select(page)
        self.app.update()
        self.assertEqual(self.app.page_title.get(), "Social exports")
        canvas = self.app._scroll_pages[str(page)]
        canvas.yview_moveto(1)
        self.app.update()
        button = self.app.social_play_button
        self.assertGreaterEqual(button.winfo_rooty(), canvas.winfo_rooty())
        self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), canvas.winfo_rooty() + canvas.winfo_height())

    def test_recognition_controls_are_reachable_in_small_session_window(self):
        page = self.app._pages["Session"]
        self.app.notebook.select(page)
        self.app.update()
        canvas = self.app._scroll_pages[str(page)]
        canvas.yview_moveto(1)
        self.app.update()
        for widget in (self.app.recognition_combo, self.app.recognition_note):
            self.assertGreaterEqual(widget.winfo_rooty(), canvas.winfo_rooty())
            self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), canvas.winfo_rooty() + canvas.winfo_height())


if __name__ == "__main__":
    unittest.main()
