"""UI contract checks using isolated jobs; no media rendering or external playback."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fpvsesh.ui import SeshApp, SOCIAL_FORMATS, write_json, read_json


class UiSettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fpvsesh-ui-test-")
        self.root = Path(self.directory.name)
        with patch.object(SeshApp, "_seed_inputs", lambda app: app._refresh_inputs()):
            self.app = SeshApp(self.root)
        self.app.attributes("-alpha", 0)
        self.app.geometry("1020x650")
        self.app.update()
        self.job = self.root / "output" / "test-job"
        self.job.mkdir(parents=True)
        self.source = self.root / "recording.mp4"
        self.source.touch()
        self.app.job_dir = self.job
        self.identity = "a" * 64
        write_json(self.job / "sources.json", [{"source": str(self.source), "duration": 30, "sha256": self.identity}])

    def tearDown(self):
        self.app.destroy()
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
        })
        self.app._restore_settings(self.job)
        self.assertFalse(self.app.settings_dirty)
        with patch.object(self.app, "_launch") as launch:
            self.app._regenerate()
        args = launch.call_args.args[0]
        for flag, expected in (("--music", str(track)), ("--music-end", "loop"), ("--social-formats", "vertical,square,portrait"),
                               ("--framing", "fill"), ("--edit-order", "chronological"), ("--recovery", "4.5"),
                               ("--duration", "180"), ("--style", "flow"), ("--codec", "h264")):
            self.assertEqual(self.value(args, flag), expected)
        self.assertIn("--no-beat-sync", args)
        self.assertNotIn("--input", args)
        self.assertNotIn("--no-music", args)
        self.assertEqual(float(self.value(args, "--focus-x")), .7)
        write_json(self.job / "settings.json", {"inputs": [str(self.source)]})
        self.app._restore_settings(self.job)
        self.assertIsNone(self.app.music_path)
        self.assertEqual(self.value(self.app._settings_args(), "--social-formats"), "none")

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


if __name__ == "__main__":
    unittest.main()
