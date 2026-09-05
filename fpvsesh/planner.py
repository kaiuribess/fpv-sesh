"""Global selection with non-overlap, diversity, and rational timeline mapping."""
from __future__ import annotations
from fractions import Fraction
import bisect
import copy
import math
import numpy as np

def overlaps(a, b, margin=.65):
    return a["identity"] == b["identity"] and max(a["start"], b["start"]) < min(a["end"], b["end"]) + margin

def near_duplicate(a, b):
    distances = [(int(x, 16) ^ int(y, 16)).bit_count() for x, y in zip(a["hash_sequence"], b["hash_sequence"])]
    return max(distances) <= 5

def plan(candidates, probes, fps, duration="auto", style="hype", overrides=None):
    target = 75 if str(duration) == "auto" else int(duration)
    by_source = {p["source"]: p for p in probes}
    for c in candidates:
        if c["source"] not in by_source:
            raise ValueError(f"Candidate source is not in this session: {c['id']}")
        start, end = float(c["start"]), float(c["end"])
        source_duration = float(by_source[c["source"]]["duration"])
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0
                or end <= start or start >= source_duration or end > source_duration + 1e-6):
            raise ValueError(f"Candidate interval is outside source bounds or empty: {c['id']}")
    overrides = overrides or {}
    keep, exclude = set(overrides.get("keep", [])), set(overrides.get("exclude", []))
    unknown = (keep | exclude) - {c["id"] for c in candidates}
    if unknown: raise ValueError(f"Override IDs no longer exist: {sorted(unknown)}")
    if keep & exclude: raise ValueError("A candidate cannot be both kept and excluded")
    chosen = []
    for c in candidates:
        c["selected"] = False
        c["selection_reason"] = "not selected: lower global rank or similar/overlapping material"
        if c["id"] in exclude: c["selection_reason"] = "excluded by user"
    for c in sorted((c for c in candidates if c["id"] in keep), key=lambda c: -c["score"]):
        if any(overlaps(c, x, 0) for x in chosen): raise ValueError("Kept candidates overlap; keep only one version of that moment")
        chosen.append(c)
    elapsed = sum(c["end"] - c["start"] for c in chosen)
    # Auto is a pacing preference. Retained maneuvers and their follow-through
    # take precedence over the nominal 75-second target, without adding filler.
    if str(duration) == "auto": target = max(target, elapsed)
    if elapsed > target + .5: raise ValueError("Kept moments exceed duration target; select a longer target")
    pool = [c for c in candidates if not c["unusable"] and c["id"] not in exclude | keep and c["score"] >= 35]
    while pool:
        available = [c for c in pool if not any(overlaps(c, x) or near_duplicate(c, x) for x in chosen) and elapsed + c["end"] - c["start"] <= target + .25]
        if not available: break
        def utility(c):
            represented = sum(x["source"] == c["source"] for x in chosen)
            similarity = max((float(np.minimum(c["hist"], x["hist"]).sum()) for x in chosen), default=0)
            return c["score"] - represented * 4 - similarity * 9
        best = max(available, key=utility)
        chosen.append(best)
        elapsed += best["end"] - best["start"]
        pool.remove(best)
    if not chosen: raise ValueError("No usable flying intervals found; review candidates and keep a moment explicitly")
    # Select distinct opener, late hero and exit. No repetitions to satisfy a target.
    ranked = sorted(chosen, key=lambda x: -x["score"])
    opener = ranked[1] if len(ranked) > 2 else ranked[0]
    hero = ranked[0]
    rest = [x for x in chosen if x is not opener and x is not hero]
    # Prefer a settled exit, including aerial pull-aways; late recording time alone
    # is not evidence of a satisfying ending and can favor rough ground arrivals.
    ending = max(rest, key=lambda x: x["score"]*.12 - x.get("end_rotation", x["rotation"]*10)*3 + (x["end"] / x["source_duration"])*2) if rest else None
    if ending: rest.remove(ending)
    ordered = [opener]
    while rest:
        previous = ordered[-1]
        build = len(ordered) / max(len(chosen), 1)
        next_clip = max(rest, key=lambda x: -abs(x["dx_in"] - previous["dx_out"]) * .8 + (x["source"] != previous["source"]) * 4 + x["score"] * build * .2)
        ordered.append(next_clip)
        rest.remove(next_clip)
    if hero is not opener: ordered.append(hero)
    if ending: ordered.append(ending)
    if "order" in overrides:
        requested_order = overrides["order"]
        if len(requested_order) != len(chosen) or set(requested_order) != {c["id"] for c in chosen}:
            raise ValueError("Editorial order must list each selected moment exactly once")
        selected_by_id = {c["id"]:c for c in chosen}
        ordered = [selected_by_id[cid] for cid in requested_order]
    out_fps = Fraction(fps)
    if out_fps <= 0:
        raise ValueError("Timeline frame rate must be positive")
    cursor, shots = 0, []
    for index, original in enumerate(ordered):
        c = copy.deepcopy(original)
        p = by_source[c["source"]]
        source_fps = Fraction(p["fps"])
        base = Fraction(p.get("time_base", "1/60"))
        origin = (Fraction(str(p["start_pts"])) * base if p.get("start_pts") is not None
                  else Fraction(str(p.get("start_time", 0))))
        if p.get("frame_pts"):
            pts = p["frame_pts"]
            start_frame = min(len(pts)-1, bisect.bisect_left(pts, (Fraction(str(c["start"])) + origin) / base))
            end_frame = min(len(pts), bisect.bisect_left(pts, (Fraction(str(c["end"])) + origin) / base))
            start = pts[start_frame] * base - origin
            # The exclusive endpoint after the final frame is the stream's
            # measured duration, not the PTS at which that last frame begins.
            end = (pts[end_frame] * base - origin if end_frame < len(pts)
                   else Fraction(str(p["duration"])))
        else:
            start_frame = max(0, round(Fraction(str(c["start"])) * source_fps))
            end_frame = min(int(p.get("frame_count") or float(source_fps) * p["duration"]), round(Fraction(str(c["end"])) * source_fps))
            start = Fraction(start_frame, 1) / source_fps
            end = Fraction(end_frame, 1) / source_fps
        frames = round((end - start) * out_fps)
        if end <= start or frames <= 0:
            raise ValueError(f"Candidate has no complete timeline frames after timestamp snapping: {c['id']}")
        if any(c["identity"] == previous["identity"]
               and max(start, Fraction(previous["source_start_time"])) <
                   min(end, Fraction(previous["source_end_time"])) for previous in shots):
            raise ValueError(f"Selected source intervals overlap after timestamp snapping: {c['id']}")
        c.update({"source_start_frame": start_frame, "source_end_frame_exclusive": end_frame,
                  "source_start_time": str(start), "source_end_time": str(end),
                  "source_pts_start": str((start + origin) / base),
                  "source_pts_end_exclusive": str((end + origin) / base),
                  "start": float(start), "end": float(end), "frames": frames,
                  "timeline_in_frame": cursor, "timeline_out_frame": cursor + frames,
                  "timeline_in": float(Fraction(cursor, 1) / out_fps), "duration": float(Fraction(frames, 1) / out_fps),
                  "role": "opener" if index == 0 else "ending" if index == len(ordered)-1 else "hero" if original is hero else "line",
                  "selected": True, "selection_reason": "reviewed editorial selection" if c.get("review_key") and c["id"] in keep else "user keep" if c["id"] in keep else "global quality, motion, variety and continuity selection",
                  "vfr": bool(p.get("frame_pts")), "rate_conversion": p["fps"] != fps or bool(p.get("frame_pts"))})
        original["selected"] = True
        original["selection_reason"] = c["selection_reason"]
        shots.append(c)
        cursor += frames
    return {"version": 1, "fps": fps, "frames": cursor, "duration": float(Fraction(cursor, 1) / out_fps),
            "target": duration, "style": style, "music": None, "music_status": "No music requested",
            "selection_method": "full-session motion/quality heuristics; maneuver names are estimates",
            "shots": shots, "warnings": ["Automatic endings retain 2.5 seconds after detected motion bursts and link bursts within that hold. These are motion estimates, not proof of complete tricks or controlled recovery; explicitly reviewed boundaries take precedence."]}
