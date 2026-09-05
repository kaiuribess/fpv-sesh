"""Real audio fixtures for local analysis, soundtrack timing, and safe mixing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
import wave

import numpy as np

from fpvsesh import music
from fpvsesh.control import Cancelled
from fpvsesh.media import locate_tools, run


def wav(path, samples, rate=48000):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


class MusicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="fpv-music-")
        cls.root = Path(cls.temp.name).resolve()
        cls.ffmpeg, cls.ffprobe = locate_tools()
        rate = 48000
        clicks = np.zeros(rate * 12, dtype=np.float32)
        pulse_time = np.arange(round(rate * .035)) / rate
        pulse = .8 * np.sin(2 * np.pi * 1800 * pulse_time) * np.exp(-pulse_time * 130)
        for start in np.arange(.25, 12, .5):
            index = round(start * rate)
            clicks[index:index + len(pulse)] += pulse
        cls.clicks = cls.root / "music café's [音楽] 🎵 & $HOME.wav"
        wav(cls.clicks, clicks)
        cls.tone = cls.root / "tone.wav"
        wav(cls.tone, .2 * np.sin(2 * np.pi * 440 * np.arange(rate * 6) / rate))
        cls.short = cls.root / "short.wav"
        wav(cls.short, .5 * np.sin(2 * np.pi * 880 * np.arange(rate * 2) / rate))
        cls.loud = cls.root / "loud.wav"
        wav(cls.loud, .98 * np.sin(2 * np.pi * 880 * np.arange(rate * 3) / rate))
        cls.silence = cls.root / "silence.wav"
        wav(cls.silence, np.zeros(rate * 3))
        cls.identities = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (cls.clicks, cls.tone, cls.short, cls.loud, cls.silence)}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.temp_case = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.temp_case.cleanup)
        self.folder = Path(self.temp_case.name)
        self.cache = self.folder / "cache"
        self.job = self.folder / "job"

    def tearDown(self):
        for path, identity in self.identities.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), identity)

    def decoded(self, path, stereo=False):
        result = subprocess.run([self.ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0",
                                 "-ac", "2", "-ar", "48000", "-f", "f32le", "-"],
                                capture_output=True, check=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        samples = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)
        return samples if stereo else samples.mean(axis=1)

    def test_clicks_have_genuine_onsets_tempo_and_offset_relative_timing(self):
        info = music.analyze_music(self.clicks, self.cache, offset=1)
        self.assertEqual(info["sha256"], self.identities[self.clicks])
        self.assertTrue(info["full_decode_validated"])
        self.assertAlmostEqual(info["duration"], 12, places=3)
        self.assertAlmostEqual(info["bpm"], 120, delta=3)
        self.assertGreater(info["confidence"], .6)
        self.assertGreaterEqual(len(info["beats"]), 19)
        for beat in info["beats"]:
            nearest = .25 + round((beat - .25) / .5) * .5
            self.assertLess(abs(beat - nearest), .045)
            self.assertTrue(0 <= beat < 11)
        json.dumps(info, allow_nan=False)

    def test_content_identity_cache_reuses_renames_but_invalidates_changed_file(self):
        source = self.folder / "replaceable.wav"
        source.write_bytes(self.clicks.read_bytes())
        first = music.analyze_music(source, self.cache)
        renamed = self.folder / "renamed.wav"
        renamed.write_bytes(source.read_bytes())
        with patch.object(music, "_execute", side_effect=AssertionError("Unexpected audio reanalysis")):
            cached = music.analyze_music(renamed, self.cache)
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(cached["path"], str(renamed.resolve()))
        source.write_bytes(self.silence.read_bytes())
        changed = music.analyze_music(source, self.cache)
        self.assertNotEqual(changed["sha256"], first["sha256"])
        self.assertFalse(changed["cache_hit"])
        self.assertEqual(changed["beats"], [])

    def test_silence_and_sustained_tone_do_not_invent_a_tempo(self):
        for source in (self.silence, self.tone):
            with self.subTest(source=source.name):
                info = music.analyze_music(source, self.cache)
                self.assertIsNone(info["bpm"])
                self.assertEqual(info["beats"], [])
                self.assertLess(info["confidence"], .2)

    def test_long_track_validation_keeps_beat_window_bounded(self):
        long_track = self.folder / "long.wav"
        wav(long_track, np.zeros(181 * 11025), rate=11025)
        info = music.analyze_music(long_track, self.cache)
        self.assertEqual(info["analysis_seconds"], 180)
        self.assertAlmostEqual(info["duration"], 181, places=3)
        self.assertTrue(info["full_decode_validated"])
        self.assertTrue(any("180 seconds" in warning for warning in info["warnings"]))

    def test_invalid_offset_missing_audio_and_bad_media_fail_clearly(self):
        for offset in (-1, float("nan"), float("inf"), 12, 13):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                music.analyze_music(self.clicks, self.cache, offset)
        corrupt = self.folder / "corrupt.mp3"
        corrupt.write_bytes(b"this is not audio")
        with self.assertRaises(RuntimeError):
            music.analyze_music(corrupt, self.cache)
        video = self.folder / "silent-video.mp4"
        run([self.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=s=64x64:r=10", "-t", "0.2",
             "-c:v", "libx264", "-an", video])
        with self.assertRaisesRegex(ValueError, "no audio stream"):
            music.analyze_music(video, self.cache)

    def test_common_formats_and_video_audio_are_decoded(self):
        for extension, codec in (("mp3", "libmp3lame"), ("m4a", "aac"), ("aac", "aac"),
                                 ("flac", "flac"), ("ogg", "libvorbis")):
            target = self.folder / f"sample.{extension}"
            run([self.ffmpeg, "-v", "error", "-i", self.short, "-c:a", codec, target])
            with self.subTest(extension=extension):
                info = music.analyze_music(target, self.cache)
                self.assertTrue(info["full_decode_validated"])
                self.assertAlmostEqual(info["duration"], 2, delta=.06)
        video = self.folder / "with-audio.mp4"
        run([self.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=s=64x64:r=10",
             "-i", self.short, "-t", "2", "-c:v", "libx264", "-c:a", "aac", video])
        self.assertTrue(music.analyze_music(video, self.cache)["full_decode_validated"])

    def test_short_music_fades_then_pads_silence_without_looping(self):
        info = music.analyze_music(self.short, self.cache)
        output = music.mix_music(None, info, self.job, 4, .5, .4)
        samples = self.decoded(output)
        self.assertAlmostEqual(len(samples) / 48000, 4, delta=.04)
        self.assertGreater(float(np.sqrt(np.mean(samples[24000:48000] ** 2))), .04)
        self.assertLess(float(np.max(np.abs(samples[round(2.2 * 48000):]))), .001)
        beginning = float(np.sqrt(np.mean(samples[:round(.04 * 48000)] ** 2)))
        middle = float(np.sqrt(np.mean(samples[24000:48000] ** 2)))
        self.assertLess(beginning, middle * .2)
        report = json.loads((self.job / "music-mix.json").read_text())
        self.assertTrue(report["short_music"])
        self.assertFalse(report["looped"])
        self.assertTrue(report["warnings"])

    def test_loop_repeats_selected_tail_and_reports_audible_join_limitation(self):
        info = music.analyze_music(self.short, self.cache, offset=.5)
        output = music.mix_music(None, info, self.job, 4, .5, .2, short_policy="loop")
        samples = self.decoded(output)
        self.assertGreater(float(np.sqrt(np.mean(samples[round(2.3 * 48000):round(2.8 * 48000)] ** 2))), .04)
        report = json.loads((self.job / "music-mix.json").read_text())
        self.assertTrue(report["looped"])
        self.assertIn("joins may be audible", report["warnings"][0])
        self.assertFalse(list(self.job.glob("music-*.loop.wav")))

    def test_mixing_preserves_both_frequencies_and_safely_replaces_canonical_source(self):
        info = music.analyze_music(self.short, self.cache)
        self.job.mkdir()
        canonical = self.job / "source-audio.m4a"
        run([self.ffmpeg, "-v", "error", "-i", self.tone, "-c:a", "aac", canonical])
        output = music.mix_music(canonical, info, self.job, 3, .4, .2)
        self.assertEqual(output, canonical)
        samples = self.decoded(output)[24000:72000]
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        self.assertGreater(spectrum[440], spectrum[600] * 100)
        self.assertGreater(spectrum[880], spectrum[600] * 100)
        tail = self.decoded(output)[round(2.3 * 48000):round(2.7 * 48000)]
        self.assertGreater(float(np.sqrt(np.mean(tail * tail))), .05)
        self.assertTrue(json.loads((self.job / "music-mix.json").read_text())["source_audio_included"])

    def test_limiter_prevents_decoded_clipping_and_zero_gain_stays_silent(self):
        info = music.analyze_music(self.loud, self.cache)
        output = music.mix_music(self.loud, info, self.job, 3, 1, .1)
        samples = self.decoded(output, stereo=True)
        self.assertLess(float(np.max(np.abs(samples))), .99)
        report = json.loads((self.job / "music-mix.json").read_text())
        self.assertLess(report["verification"]["sample_peak_dbfs"], -.1)
        output = music.mix_music(None, info, self.job, 3, 0, .1)
        self.assertLess(float(np.max(np.abs(self.decoded(output)))), 1e-7)

    def test_music_change_or_output_collision_is_rejected_before_overwrite(self):
        mutable = self.folder / "music.wav"
        mutable.write_bytes(self.short.read_bytes())
        info = music.analyze_music(mutable, self.cache)
        mutable.write_bytes(self.silence.read_bytes())
        with self.assertRaisesRegex(ValueError, "changed after analysis"):
            music.mix_music(None, info, self.job, 3, .4, .2)
        self.job.mkdir(exist_ok=True)
        canonical = self.job / "source-audio.m4a"
        run([self.ffmpeg, "-v", "error", "-i", self.short, "-c:a", "aac", canonical])
        info = music.analyze_music(canonical, self.cache)
        before = canonical.read_bytes()
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            music.mix_music(None, info, self.job, 3, .4, .2)
        self.assertEqual(canonical.read_bytes(), before)

    def test_cancel_keeps_existing_soundtrack_and_cleans_owned_partials(self):
        info = music.analyze_music(self.short, self.cache)
        self.job.mkdir()
        canonical = self.job / "source-audio.m4a"
        canonical.write_bytes(b"previous completed soundtrack")
        real_execute = music._execute
        def cancel_after_encoding(command, checkpoint, **kwargs):
            result = real_execute(command, checkpoint, **kwargs)
            if str(command[-1]).endswith(".partial.m4a"):
                raise Cancelled("cancel after generated audio, before replacement")
            return result
        with patch.object(music, "_execute", side_effect=cancel_after_encoding):
            with self.assertRaises(Cancelled):
                music.mix_music(None, info, self.job, 3, .4, .2)
        self.assertEqual(canonical.read_bytes(), b"previous completed soundtrack")
        self.assertFalse(list(self.job.glob("music-*.partial.m4a")))
        self.assertFalse((self.job / "music-mix.json").exists())
        with self.assertRaises(Cancelled):
            music.analyze_music(self.clicks, self.folder / "cancel-cache", checkpoint=lambda: (_ for _ in ()).throw(Cancelled()))

    def test_running_native_decode_is_terminated_by_cooperative_cancel(self):
        processes = []
        original_popen = subprocess.Popen
        def start(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append(process)
            return process
        began = time.monotonic()
        def checkpoint():
            if time.monotonic() - began > .15:
                raise Cancelled("stop running decode")
        with patch.object(music.subprocess, "Popen", side_effect=start):
            with self.assertRaises(Cancelled):
                music._execute([self.ffmpeg, "-v", "error", "-re", "-i", self.tone,
                                "-map", "0:a:0", "-f", "null", "-"], checkpoint)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertLess(time.monotonic() - began, 2)


if __name__ == "__main__":
    unittest.main()
