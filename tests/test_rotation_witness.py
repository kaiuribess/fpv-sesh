"""Real sampled-video rotation evidence and tracking-loss abstention."""
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from fpvsesh.control import Cancelled
from fpvsesh.media import locate_tools
from fpvsesh.rotation_witness import inspect_rotation


class RotationWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg, _ = locate_tools()
        cls.temp = tempfile.TemporaryDirectory(prefix="rotation-witness-")
        cls.folder = Path(cls.temp.name)
        generator = np.random.default_rng(804)
        cls.texture = cv2.GaussianBlur(generator.integers(20, 236, (512, 512), dtype=np.uint8), (3, 3), 0)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def video(self, name, angles, blank=(), noise=()):
        path = self.folder / (name+".avi")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 30, (320, 320), False)
        self.assertTrue(writer.isOpened())
        generator = np.random.default_rng(19)
        try:
            for index, angle in enumerate(angles):
                if index in blank:
                    frame = np.zeros((320, 320), np.uint8)
                elif index in noise:
                    frame = generator.integers(0, 256, (320, 320), dtype=np.uint8)
                else:
                    affine = cv2.getRotationMatrix2D((256, 256), angle, 1)
                    frame = cv2.warpAffine(self.texture, affine, (512, 512), flags=cv2.INTER_LINEAR)[96:416, 96:416]
                writer.write(frame)
        finally:
            writer.release()
        return path

    def inspect(self, path, frames):
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        result = inspect_rotation(path, 0, frames/30, self.ffmpeg)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertTrue(result["full_source_window"])
        self.assertNotIn("confidence", result)
        self.assertNotIn("flip_count", result)
        return result

    def test_full_continuous_rotation_is_witnessed_with_quiet_exit(self):
        angles = [0]*15+list(np.linspace(0, 360, 61))+[360]*44
        result = self.inspect(self.video("full-roll", angles), len(angles))
        self.assertTrue(result["complete_image_rotation"], result)
        full = [burst for burst in result["bursts"] if burst["complete_image_rotation"]]
        self.assertEqual(len(full), 1)
        self.assertTrue(330 <= abs(full[0]["signed_degrees"]) <= 390)
        self.assertGreater(full[0]["after_seconds"], 1)
        self.assertGreater(result["valid_fraction"], .9)

    def test_ordinary_bank_and_frozen_tilt_do_not_become_full_rotations(self):
        for name, angles in (("frozen-bank", [20]*60),
                             ("bank-turn", list(np.linspace(0, 20, 30))+list(np.linspace(20, 0, 30)))):
            result = self.inspect(self.video(name, angles), len(angles))
            self.assertFalse(result["complete_image_rotation"], result)

    def test_direction_reversal_does_not_add_opposite_rotations(self):
        angles = list(np.linspace(0, 200, 41))+list(np.linspace(200, 0, 41))
        result = self.inspect(self.video("reverse-direction", angles), len(angles))
        self.assertFalse(result["complete_image_rotation"], result)
        self.assertTrue(any(burst["signed_degrees"] > 0 for burst in result["bursts"]))
        self.assertTrue(any(burst["signed_degrees"] < 0 for burst in result["bursts"]))

    def test_tracking_gap_never_completes_rotation_across_missing_evidence(self):
        angles = list(np.linspace(0, 360, 73))+[360]*20
        result = self.inspect(self.video("obscured-roll", angles, blank=range(33, 41)), len(angles))
        self.assertFalse(result["complete_image_rotation"], result)
        self.assertLess(result["valid_fraction"], 1)

    def test_independent_frame_noise_abstains(self):
        angles = [0]*40
        result = self.inspect(self.video("noise", angles, noise=range(40)), len(angles))
        self.assertFalse(result["complete_image_rotation"])
        self.assertEqual(result["status"], "unmeasured")
        self.assertEqual(result["valid_fraction"], 0)

    def test_source_bounds_and_cancellation(self):
        source = self.video("short-source", [0]*30)
        with self.assertRaisesRegex(ValueError, "ended early"):
            inspect_rotation(source, 0, 2, self.ffmpeg)
        def cancel():
            raise Cancelled("Stopped at boundary")
        with self.assertRaises(Cancelled):
            inspect_rotation(source, 0, 1, self.ffmpeg, cancel)
        for start, end in ((-1, 2), (0, 13), (True, 1), (0, math.nan)):
            with self.assertRaises(ValueError):
                inspect_rotation(source, start, end, self.ffmpeg)


if __name__ == "__main__":
    unittest.main()
