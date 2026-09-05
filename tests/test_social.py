"""Decode real social outputs and check framing, timing, audio and safe reuse."""
from fractions import Fraction
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fpvsesh import media, social


class SocialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg, _ = media.locate_tools()
        cls.temp = tempfile.TemporaryDirectory(prefix="fpvsesh-social-")
        cls.folder = Path(cls.temp.name)
        cls.source = cls.folder / "flight café's [1] & $source.mp4"
        media.run([cls.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=30000/1001",
                   "-vf", "drawbox=x=0:y=0:w=30:h=90:color=red:t=fill,drawbox=x=130:y=0:w=30:h=90:color=blue:t=fill",
                   "-frames:v", "24", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "10",
                   "-pix_fmt", "yuv420p", "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709", str(cls.source)])
        cls.metadata = media.probe(cls.source)
        cls.original_hash = media.sha256_file(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def timeline(self, frames=5, two=True, metadata=None):
        meta = metadata or self.metadata
        fps = Fraction(meta["fps"])
        shots = []
        for index, start_frame in enumerate([1, 12] if two else [1]):
            start, end = Fraction(start_frame, 1) / fps, Fraction(start_frame + frames, 1) / fps
            shots.append({"source": meta["source"], "identity": meta["identity"],
                          "start": float(start), "end": float(end), "duration": float(Fraction(frames, 1) / fps),
                          "source_start_time": str(start), "source_end_time": str(end),
                          "source_start_frame": start_frame, "source_end_frame_exclusive": start_frame+frames,
                          "frames": frames, "role": "opener" if index==0 else "ending"})
        count = frames * len(shots)
        return {"fps": str(fps), "frames": count, "duration": float(Fraction(count, 1)/fps), "shots": shots}

    def settings(self, framing="blur", formats=None, focus=.5):
        return {"social_formats": formats or ["vertical", "square", "portrait"], "framing": framing,
                "focus_x": focus, "look": "natural", "strength": 0}

    def job(self, name, timeline, audio=True):
        folder = self.folder / name
        folder.mkdir(exist_ok=True)
        if audio:
            media.run([self.ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000",
                       "-t", f"{timeline['duration']:.9f}", "-ac", "2", "-c:a", "aac", "-b:a", "192k", str(folder/"source-audio.m4a")])
        return folder

    def render(self, timeline, settings, name, *, preview=True, audio=True, checkpoint=lambda:None, metadata=None):
        job = self.job(name, timeline, audio)
        events = []
        result = social.export_social(timeline, [metadata or self.metadata], settings, job, self.folder/"cache",
                                      lambda *args: events.append(args), checkpoint, preview)
        self.assertEqual(events[-1][1], 1)
        return result

    def first_frame(self, path):
        meta = media.probe(path, include_hash=False)
        process = subprocess.run([self.ffmpeg, "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo",
                                  "-pix_fmt", "rgb24", "-"], capture_output=True, check=True,
                                 creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        return np.frombuffer(process.stdout, dtype=np.uint8).reshape(meta["height"], meta["width"], 3)

    def test_plan_sizes_invalid_inputs_and_platform_warnings(self):
        plan = social.social_export_plan(self.settings(), {"fps":"120/1","duration":181})
        self.assertEqual((plan["vertical"]["width"],plan["vertical"]["height"]),(1080,1920))
        self.assertEqual((plan["portrait"]["width"],plan["portrait"]["height"]),(1080,1350))
        self.assertTrue(any("three-minute" in w for w in plan["square"]["warnings"]))
        self.assertTrue(any("23" in w for w in plan["square"]["warnings"]))
        for settings in ({"social_formats":["unknown"]}, {"social_formats":"vertical"},
                         {"social_formats":["vertical"],"focus_x":float("nan")},
                         {"social_formats":["vertical"],"framing":"tracking"}):
            with self.assertRaises(ValueError):
                social.social_export_plan(settings)
        self.assertEqual(social.social_export_plan({"social_formats":[]}), {})

    def test_all_profiles_and_framings_decode_with_exact_timing_and_audio(self):
        timeline = self.timeline()
        frozen = copy.deepcopy(timeline)
        for framing in ("blur","fit","fill"):
            with self.subTest(framing=framing):
                results = self.render(timeline,self.settings(framing),"profiles-"+framing)
                for code, result in results.items():
                    verify = result["verification"]
                    self.assertTrue(verify["passed"], verify["errors"])
                    self.assertEqual(verify["timestamps"]["frame_count"],10)
                    self.assertEqual(Fraction(verify["probe"]["fps"]),Fraction(30000,1001))
                    self.assertEqual(verify["probe"]["pix_fmt"],"yuv420p")
                    self.assertEqual(verify["probe"]["codec"],"h264")
                    self.assertEqual(verify["probe"]["audio_streams"][0]["sample_rate"],"48000")
                    self.assertTrue(verify["social"]["faststart"])
                    self.assertFalse(verify["black_intervals"])
                    self.assertTrue(Path(result["poster"]).is_file())
                    self.assertEqual([r["frames"] for r in result["records"]],[5,5])
                    image = self.first_frame(result["path"])
                    self.assertGreater(float(image[image.shape[0]//2].mean()),20)
                    if framing=="fit":
                        self.assertLess(float(image[:8].mean()),4)
                    else:
                        self.assertGreater(float(image[:8].mean()),8)
        self.assertEqual(timeline,frozen)
        self.assertEqual(media.sha256_file(self.source),self.original_hash)

    def test_fill_focus_keeps_requested_side_and_blur_keeps_both_edges(self):
        timeline = self.timeline(frames=3,two=False)
        left = self.render(timeline,self.settings("fill",["vertical"],0),"focus-left")["vertical"]
        right = self.render(timeline,self.settings("fill",["vertical"],1),"focus-right")["vertical"]
        a,b = self.first_frame(left["path"]),self.first_frame(right["path"])
        self.assertGreater(float(a[:,10:80,0].mean()),float(a[:,10:80,2].mean())+80)
        self.assertGreater(float(b[:,-80:-10,2].mean()),float(b[:,-80:-10,0].mean())+80)
        result = self.render(timeline,self.settings("blur",["vertical"]),"whole-frame")["vertical"]
        c = self.first_frame(result["path"])
        center = c[c.shape[0]//2-10:c.shape[0]//2+10]
        self.assertGreater(float(center[:,5:45,0].mean()),float(center[:,5:45,2].mean())+80)
        self.assertGreater(float(center[:,-45:-5,2].mean()),float(center[:,-45:-5,0].mean())+80)

    def test_final_resolution_and_real_encoder_output(self):
        result = self.render(self.timeline(frames=1,two=False),self.settings("blur",["vertical"]),
                             "one-frame-final",preview=False,audio=False)["vertical"]
        meta = result["verification"]["probe"]
        self.assertEqual((meta["width"],meta["height"]),(1080,1920))
        self.assertTrue(result["verification"]["passed"])
        self.assertEqual(result["verification"]["social"]["audio_mode"],"silence")

    def test_nvenc_failure_uses_verified_software_fallback(self):
        original_run = social.run
        def without_nvenc(cmd, *args, **kwargs):
            if "h264_nvenc" in cmd:
                return subprocess.CompletedProcess(cmd,1,"","simulated missing NVENC")
            return original_run(cmd,*args,**kwargs)
        with patch.object(social,"run",side_effect=without_nvenc):
            result = self.render(self.timeline(frames=1,two=False),self.settings("fit",["square"]),
                                 "fallback",preview=False,audio=False)["square"]
        self.assertEqual(result["records"][0]["encoder"],"libx264")
        self.assertEqual(result["verification"]["probe"]["color_primaries"],"bt709")
        self.assertTrue(result["records"][0]["failures"])

    def test_cache_reuse_and_tampered_cache_rebuild(self):
        timeline = self.timeline(frames=2,two=False)
        settings = self.settings("fit",["portrait"])
        first = self.render(timeline,settings,"cache-first")["portrait"]
        cached = Path(first["records"][0]["output_path"])
        original_mtime = cached.stat().st_mtime_ns
        second = self.render(timeline,settings,"cache-second")["portrait"]
        self.assertEqual(cached.stat().st_mtime_ns,original_mtime)
        self.assertEqual(first["records"][0]["_cache_sha256"],second["records"][0]["_cache_sha256"])
        with cached.open("ab") as stream:
            stream.write(b"changed application cache")
        third = self.render(timeline,settings,"cache-third")["portrait"]
        self.assertTrue(third["verification"]["passed"])
        self.assertEqual(media.sha256_file(cached),third["records"][0]["_cache_sha256"])

    def test_checkpoint_cancellation_cleans_only_partial_files(self):
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InterruptedError("cancelled by user")
        with self.assertRaises(InterruptedError):
            self.render(self.timeline(frames=4,two=False),self.settings("fill",["portrait"],.15),
                        "cancelled",checkpoint=cancel)
        self.assertFalse(list((self.folder/"cache").rglob("*.partial.mp4")))
        self.assertFalse((self.folder/"cancelled/social-preview/portrait.mp4").exists())
        self.assertEqual(media.sha256_file(self.source),self.original_hash)

    def test_rotated_non_square_pixels_are_contained_without_stretching(self):
        fixture = self.folder/"anamorphic.mp4"
        rotated = self.folder/"rotated.mp4"
        media.run([self.ffmpeg,"-v","error","-f","lavfi","-i","color=c=yellow:s=160x90:r=30",
                   "-vf","setsar=2/1","-frames:v","8","-c:v","libx264","-pix_fmt","yuv420p",str(fixture)])
        media.run([self.ffmpeg,"-v","error","-display_rotation:v:0","90","-i",str(fixture),"-c","copy",str(rotated)])
        metadata = media.probe(rotated)
        self.assertEqual(abs(metadata["rotation"]),90)
        result = self.render(self.timeline(frames=2,two=False,metadata=metadata),self.settings("fit",["square"]),
                             "rotated",metadata=metadata)["square"]
        frame = self.first_frame(result["path"])
        mask = frame.mean(axis=2)>30
        x = np.flatnonzero(mask.any(axis=0))
        y = np.flatnonzero(mask.any(axis=1))
        # Original DAR32:9, rotated to9:32; contained in360-square =>100×360.
        self.assertLessEqual(abs((x[-1]-x[0]+1)-100),2)
        self.assertEqual(y[-1]-y[0]+1,360)


if __name__ == "__main__":
    unittest.main()
