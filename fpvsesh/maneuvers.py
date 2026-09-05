"""Temporal windows and measured support for tentative video-model labels.

This module is not a learned trick classifier. In particular, summing image
rotation does not count physical rolls or flips. The existing analysis uses
six frame pairs per second and may lose tracks during fast acrobatics.
"""
from __future__ import annotations

import math
from numbers import Real
from statistics import median


TRICK_LABELS = (
    "ordinary flight", "tree weaving", "roll", "flip", "split-S", "powerloop",
    "dive", "orbit", "landing", "crash", "uncertain",
)

# These descriptions guide a separate internet-pretrained video model. They
# are a vocabulary, not training examples or proof that a model learned FPV.
TRICK_DEFINITIONS = {
    "ordinary flight": "Continuous travel or ordinary turns without a clear complete acrobatic maneuver.",
    "tree weaving": "A continuing flight line alternates around visible trees; vegetation alone is insufficient.",
    "roll": "A complete rotation about the drone's forward axis. An ordinary bank or incomplete rotation is insufficient.",
    "flip": "A complete end-over-end pitch rotation with visible entry and exit; fast vertical image motion alone is insufficient.",
    "split-S": "A half roll into inversion followed by a descending half loop, exiting toward the opposite direction.",
    "powerloop": "A looping flight path over or around an obstacle and back through the approach area; a pitch flick alone is insufficient.",
    "dive": "A sustained nose-down descent with surrounding geometry supporting descent; merely seeing ground is insufficient.",
    "orbit": "A flight path circles a visible stationary subject; image spin alone is insufficient.",
    "landing": "An approach ends in sustained settled contact. Hovering and intentional perching can look similar.",
    "crash": "Visible unintended impact or loss of control; inversion, motion blur, or a stationary ending alone is insufficient.",
    "uncertain": "The sequence is ambiguous, obscured, incomplete, or does not provide enough evidence for another label.",
}

TAXONOMY_SOURCES = (
    {"title": "Rotor Riot FPV Freestyle Tricktionary",
     "url": "https://rotorriot.com/blogs/tutorials-guides/fpv-freestyle-tricktionary",
     "scope": "FPV rotation families, object-dependent maneuvers, orbits, dives, and intentional perches"},
    {"title": "ArduPilot Flip Mode",
     "url": "https://ardupilot.org/copter/docs/flip-mode.html",
     "scope": "Roll/pitch axes and the distinction between rotation and recovered entry attitude"},
    {"title": "Betaflight 4.3 Tuning Notes",
     "url": "https://betaflight.com/docs/wiki/tuning/4-3-Tuning-Notes",
     "scope": "Angular rates; 670 degrees per second demonstrates why sparse frames can miss acrobatics"},
    {"title": "OpenCV estimateAffinePartial2D",
     "url": "https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html",
     "scope": "Image-plane translation, rotation, and uniform scale do not recover physical flight axes"},
)


def _number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _bounds(start, end):
    if not (_number(start) and _number(end) and 0 <= start < end):
        raise ValueError("A measured interval needs finite bounds with 0 <= start < end")
    return float(start), float(end)


def build_windows(analysis, mode="auto"):
    """Cover the complete source with chronological overlapping model windows.

    Auto uses 8-second windows with 1-second overlap. Thorough uses 6-second
    windows with 2-second overlap. Tail windows stop at the actual source end.
    Estimated scene cuts do not split windows: a fast trick can resemble a cut.
    Overlap helps boundary context but cannot prove a whole maneuver was seen.
    """
    if mode not in {"auto", "off", "thorough"}:
        raise ValueError("Trick analysis mode must be auto, thorough, or off")
    duration = analysis.get("duration") if isinstance(analysis, dict) else None
    if not _number(duration) or duration < 0:
        raise ValueError("Flight duration must be a finite nonnegative number")
    if mode == "off" or duration == 0:
        return []
    width, stride = (8, 7) if mode == "auto" else (6, 4)
    # Each start is computed from an integer index, avoiding accumulated drift.
    windows, index = [], 0
    while True:
        start = float(index * stride)
        end = min(start + width, float(duration))
        windows.append({"start": start, "end": end})
        if end >= duration:
            return windows
        index += 1


def motion_evidence(rows, start, end):
    """Describe existing image-motion evidence without assigning a trick.

    Rows are endpoint measurements for frame pairs. Only pairs wholly inside
    the requested interval are accumulated. Motion thresholds are expressed
    per second (equivalent to the existing six-fps proposal thresholds).
    Tracking failures explicitly marked rotation_valid/motion_valid=False are
    excluded. Legacy rows lack these flags; their quality remains unverified.
    Never interpret the returned rotation sum as a physical turn count.
    """
    start, end = _bounds(start, end)
    indexed, duplicates, malformed = {}, set(), 0
    for row in rows or []:
        if not isinstance(row, dict) or not _number(row.get("t")) or row["t"] < 0:
            malformed += 1
            continue
        t = float(row["t"])
        if t in indexed:
            duplicates.add(t)
        else:
            indexed[t] = row
    ordered = sorted(indexed.items())
    chosen = [(t, row) for t, row in ordered if start <= t < end]
    rotations, rates, speeds, proximity, deltas = [], [], [], [], []
    valid_flags, unverified_pairs, rejected_pairs = 0, 0, 0
    last_sign, sign_changes = None, 0
    previous = None
    for t, row in ordered:
        if start <= t < end:
            prior_t = previous[0] if previous is not None else None
            # Missing frames and ambiguous duplicate times break the chain.
            valid_pair = (prior_t is not None and prior_t >= start and
                          t not in duplicates and prior_t not in duplicates)
            if valid_pair:
                dt = t - prior_t
                deltas.append(dt)
                valid_pair = 0 < dt <= .5
            if not valid_pair:
                rejected_pairs += 1
                last_sign = None
            else:
                flags = [row.get(key) for key in ("rotation_valid", "motion_valid")]
                valid_flags += sum(isinstance(value, bool) for value in flags)
                if not all(isinstance(value, bool) for value in flags):
                    unverified_pairs += 1
                rotation, motion = row.get("rotation"), row.get("motion")
                if _number(rotation) and row.get("rotation_valid") is not False:
                    rotations.append(float(rotation))
                    rate = float(rotation) / dt
                    rates.append(rate)
                    if abs(rate) >= 48 - 1e-6:
                        sign = 1 if rate > 0 else -1
                        sign_changes += last_sign is not None and last_sign != sign
                        last_sign = sign
                else:
                    last_sign = None
                if _number(motion) and motion >= 0 and row.get("motion_valid") is not False:
                    speeds.append(float(motion) / dt)
                near, fit = row.get("proximity"), row.get("parallax_confidence")
                if (_number(near) and _number(fit) and 0 <= near <= 1 and 0 <= fit <= 1
                        and row.get("motion_valid") is not False):
                    proximity.append(float(near * fit))
        previous = (t, row)

    # These thresholds create motion proposals only. A bank can exceed the
    # rotation threshold; textured grass can exceed the close-pass threshold.
    speed = median(speeds) if speeds else None
    stationary = sum(value < 2.4 - 1e-6 for value in speeds) / len(speeds) if speeds else None
    limits = [
        "Image-plane motion supports a tentative video interpretation; it cannot establish physical roll/pitch axes or count flips.",
        "Low motion cannot distinguish hovering, landing, an intentional perch, or a stopped recording.",
        "A complete maneuver and continued flight must be visible before describing a successful exit.",
    ]
    if unverified_pairs:
        limits.append("Legacy motion rows have no tracking-validity flags; fit quality and losses are unknown.")
    if rejected_pairs:
        limits.append("Pairs crossing the interval boundary, missing-frame gaps, or duplicate timestamps were excluded.")
    if malformed:
        limits.append("Malformed motion rows were ignored.")
    if deltas and median(deltas) >= 1 / 12 - 1e-6:
        limits.append("Sparse temporal sampling can miss fast rotations and recovery between samples.")
    measured = bool(speeds or rotations)
    return {
        "status": "measured" if measured else "unmeasured",
        "sample_count": len(chosen),
        "motion_sample_count": len(speeds),
        "rotation_sample_count": len(rotations),
        "sampling_fps_estimate": round(1 / median(deltas), 3) if deltas else None,
        "maximum_sample_gap_seconds": round(max(deltas), 6) if deltas else None,
        "tracking_quality": "unverified" if unverified_pairs else "reported" if valid_flags else "unavailable",
        "rotation_present": any(abs(value) >= 48 - 1e-6 for value in rates) if rates else None,
        "close_pass_present": sum(proximity) / len(proximity) >= .28 if proximity else None,
        "stationary_fraction": round(stationary, 4) if stationary is not None else None,
        "rotation_sign_changes": sign_changes if rates else None,
        "signed_image_rotation_degrees": round(sum(rotations), 3) if rotations else None,
        "absolute_image_rotation_degrees": round(sum(abs(value) for value in rotations), 3) if rotations else None,
        "peak_image_rotation_degrees_per_second": round(max(abs(value) for value in rates), 3) if rates else None,
        "median_motion_pixels_per_second": round(speed, 3) if speed is not None else None,
        "motion_level": "unknown" if speed is None else "low" if speed < 2.4 - 1e-6 else "high" if speed >= 48 - 1e-6 else "moving",
        "limitations": limits,
    }
