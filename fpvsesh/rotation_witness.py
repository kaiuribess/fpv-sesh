"""Conservative 30-fps image-plane rotation witnesses, not physical trick labels.

A continuous image rotation can corroborate an uncertain video interpretation.
It cannot distinguish roll/pitch/yaw in the drone frame, prove airborne flight,
or certify that a recovery avoided a crash.
"""
from __future__ import annotations

import math
import os
import subprocess
import time

import cv2
import numpy as np

SAMPLE_FPS = 30
CANVAS = 320
MAX_SECONDS = 12
VERSION = 1


def _decode(source, start, end, ffmpeg, checkpoint):
    expected = math.ceil((end-start)*SAMPLE_FPS-1e-6)
    if expected < 2:
        raise ValueError("Rotation observation needs at least two sampled frames")
    # Source-time trimming excludes later content without output -t rounding
    # away the last legitimate frame at fractional source endpoints.
    filters = (f"trim=duration={end-start},fps={SAMPLE_FPS}:start_time=0:eof_action=pass,"
               f"scale={CANVAS}:{CANVAS}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
               f"pad={CANVAS}:{CANVAS}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    args = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", str(start),
            "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", filters, "-frames:v", str(expected), "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    began = time.monotonic()
    try:
        while True:
            checkpoint()
            try:
                raw, error = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic()-began > 120:
                    raise RuntimeError("Timed out decoding a rotation observation")
        if process.returncode:
            raise RuntimeError(error.decode("utf-8", errors="replace")[-1000:])
        count, remainder = divmod(len(raw), CANVAS*CANVAS)
        if remainder or count != expected:
            raise ValueError(f"Rotation window ended early: expected {expected} frames, decoded {count}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(count, CANVAS, CANVAS)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


def _pair(previous, current):
    """Return angle and fit diagnostics only when a distributed fit survives."""
    mask = np.zeros_like(previous)
    mask[16:-16, 16:-16] = 255
    # Dark padding, prop corners, and a single moving patch must not anchor a
    # global camera-rotation claim. Real dark scenes may conservatively abstain.
    mask[previous < 8] = 0
    points = cv2.goodFeaturesToTrack(previous, maxCorners=300, qualityLevel=.015,
                                    minDistance=7, blockSize=5, mask=mask)
    if points is None or len(points) < 30:
        return None
    options = {"winSize": (25, 25), "maxLevel": 3,
               "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}
    destination, forward, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
    if destination is None:
        return None
    back, backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, destination, None, **options)
    if back is None:
        return None
    errors = np.linalg.norm(back-points, axis=2).ravel()
    valid = ((forward.ravel() == 1) & (backward.ravel() == 1) & np.isfinite(errors) &
             (errors <= 1.0) & np.isfinite(destination).all(axis=(1, 2)))
    if valid.sum() < 30:
        return None
    source = points[valid].reshape(-1, 2)
    target = destination[valid].reshape(-1, 2)
    affine, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                               ransacReprojThreshold=2, maxIters=1500, confidence=.99)
    if affine is None or inliers is None or not np.isfinite(affine).all():
        return None
    selected = inliers.ravel() == 1
    count = int(selected.sum())
    if count < 30 or count/len(source) < .7:
        return None
    source_inliers = source[selected]
    predicted = source_inliers @ affine[:, :2].T + affine[:, 2]
    residual = float(np.median(np.linalg.norm(predicted-target[selected], axis=1)))
    spread = float(cv2.contourArea(cv2.convexHull(source_inliers)) / previous.size)
    quadrants = {(int(x >= CANVAS/2), int(y >= CANVAS/2)) for x, y in source_inliers}
    scale = float(math.hypot(affine[0, 0], affine[1, 0]))
    angle = math.degrees(math.atan2(affine[1, 0], affine[0, 0]))
    if residual > 1.0 or spread < .15 or len(quadrants) < 3 or not .94 <= scale <= 1.06 or abs(angle) > 35:
        return None
    return {"degrees": float(angle), "inliers": count, "spread": spread,
            "residual_pixels": residual, "scale": scale}


def _summarize(measurements, start, end, sampled_frames):
    bursts, active = [], None
    dt = 1/SAMPLE_FPS

    def finish():
        nonlocal active
        if active is not None and abs(active["signed_degrees"]) >= 10:
            active["signed_degrees"] = round(active["signed_degrees"], 2)
            active["complete_image_rotation"] = abs(active["signed_degrees"]) >= 330
            bursts.append(active)
        active = None

    for index, measurement in enumerate(measurements, start=1):
        t = start+index*dt
        if measurement is None:
            finish()
            continue
        angle = measurement["degrees"]
        if abs(angle) >= 1:
            sign = 1 if angle > 0 else -1
            if active is not None and sign != (1 if active["signed_degrees"] > 0 else -1):
                finish()
            if active is None:
                active = {"start": round(t-dt, 6), "end": round(t, 6), "signed_degrees": 0.0}
            active["signed_degrees"] += angle
            active["end"] = round(t, 6)
        elif active is not None:
            if t-active["end"] >= .2-1e-6:
                finish()
            elif angle*active["signed_degrees"] > 0:
                active["signed_degrees"] += angle
    finish()

    for burst in bursts:
        after = 0.0
        # Quiet valid image rotation may be a static view. This duration is
        # deliberately never called successful recovery or controlled flight.
        first = max(0, round((burst["end"]-start)*SAMPLE_FPS))
        for measurement in measurements[first:]:
            if measurement is None or abs(measurement["degrees"]) >= 1:
                break
            after += dt
        burst["after_seconds"] = round(min(after, end-burst["end"]), 3)
    valid = [measurement for measurement in measurements if measurement is not None]
    fraction = len(valid)/len(measurements) if measurements else 0.0
    complete = any(burst["complete_image_rotation"] for burst in bursts)
    reason = ("A continuous image-plane rotation exceeded 330 degrees on distributed, validated tracks."
              if complete else "No complete image-plane rotation was measured in an unbroken valid track sequence."
              if valid else "Tracking did not provide reliable distributed frame-pair measurements.")
    return {"version": VERSION, "start": start, "end": end, "sample_fps": SAMPLE_FPS,
            "sampled_frames": sampled_frames, "coverage_seconds": min(end-start, sampled_frames/SAMPLE_FPS),
            "full_source_window": True, "status": "measured" if fraction >= .8 else "partial" if valid else "unmeasured",
            "valid_fraction": round(fraction, 4), "valid_pairs": len(valid), "total_pairs": len(measurements),
            "complete_image_rotation": complete, "bursts": bursts, "reason": reason,
            "fit_summary": {"median_inliers": round(float(np.median([item["inliers"] for item in valid])), 1) if valid else None,
                            "median_residual_pixels": round(float(np.median([item["residual_pixels"] for item in valid])), 3) if valid else None},
            "method": "30-fps image-plane tracking witness; not a learned physical trick classifier",
            "limitations": ["Image rotation does not identify the drone's physical roll, pitch, or yaw axis.",
                            "after_seconds measures valid quiet image rotation, not airborne recovery or freedom from a crash.",
                            "Failed tracking or a direction reversal breaks accumulation; fast blur and poorly textured views may abstain.",
                            "A bank, tracking orbit, or externally rotating camera can produce related image motion."]}


def inspect_rotation(source, start, end, ffmpeg, checkpoint=lambda: None):
    """Decode a complete bounded source window and measure robust rotation."""
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (start, end)) or not 0 <= start < end or end-start > MAX_SECONDS):
        raise ValueError("Choose a finite rotation window between two frames and 12 seconds")
    checkpoint()
    frames = _decode(source, float(start), float(end), ffmpeg, checkpoint)
    measurements = []
    old_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(2)
        for previous, current in zip(frames, frames[1:]):
            checkpoint()
            measurements.append(_pair(previous, current))
    finally:
        cv2.setNumThreads(old_threads)
    checkpoint()
    return _summarize(measurements, float(start), float(end), len(frames))
