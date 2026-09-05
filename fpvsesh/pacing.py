"""Conservative music timing: extend safe exits; never shorten a maneuver."""
from __future__ import annotations

import bisect
import copy
from fractions import Fraction
import math

from .analysis import guard_recovery_end


def favor_beats(timeline, probes, analyses, music, recovery=2.5, max_extension=.8):
    result = copy.deepcopy(timeline)
    result["beat_timing"] = {"aligned_cuts": 0, "extended_seconds": 0.0,
                             "policy": "Extend safe automatic exits only; never trim tricks or exact reviewed passages"}
    beats = sorted({float(b) for b in music.get("beats", []) if math.isfinite(float(b)) and b >= 0})
    if not beats or float(music.get("confidence", 0)) < .2:
        result["beat_timing"]["status"] = "No sufficiently clear beat grid; original flight timing retained"
        return result
    by_source = {p["source"]: p for p in probes}
    rows_by_source = {a["source"]: a for a in analyses}
    fps = Fraction(result["fps"])
    total_frames = result["frames"]
    cap = None if str(result["target"]) == "auto" else round(int(result["target"]) * fps)
    cursor = 0
    extended_frames = 0
    for index, shot in enumerate(result["shots"]):
        boundary = float(Fraction(cursor + shot["frames"], 1) / fps)
        protected = bool(shot.get("review_key") or shot.get("user_kept") or
                         "user keep" in shot.get("selection_reason", "") or
                         "reviewed" in shot.get("selection_reason", ""))
        if not protected and index < len(result["shots"]) - 1:
            source = by_source[shot["source"]]
            analysis = rows_by_source.get(shot["source"], {})
            rows = analysis.get("rows", [])
            arrival = analysis.get("terminal_arrival")
            source_limit = min(source["duration"] - .1,
                               arrival["exclude_start"] if arrival else source["duration"])
            for beat in beats[bisect.bisect_left(beats, boundary):]:
                extra = beat - boundary
                if extra > max_extension:
                    break
                if extra < .5 / float(fps):
                    result["beat_timing"]["aligned_cuts"] += 1
                    break
                proposed = shot["end"] + extra
                if not rows or proposed > source_limit:
                    continue
                source_fps = Fraction(source["fps"])
                base = Fraction(source.get("time_base", "1/60"))
                origin = (Fraction(str(source["start_pts"])) * base if source.get("start_pts") is not None
                          else Fraction(str(source.get("start_time", 0))))
                if source.get("frame_pts"):
                    pts = source["frame_pts"]
                    end_frame = min(len(pts), bisect.bisect_left(pts, (Fraction(str(proposed)) + origin) / base))
                    end = pts[end_frame] * base - origin if end_frame < len(pts) else Fraction(str(source["duration"]))
                else:
                    end_frame = min(int(source.get("frame_count") or source["duration"] * float(source_fps)),
                                    round(Fraction(str(proposed)) * source_fps))
                    end = Fraction(end_frame, 1) / source_fps
                guarded = guard_recovery_end(rows, shot["start"], float(end), source_limit, recovery)
                if guarded is None or guarded > float(end) + 1e-6 or end <= Fraction(shot["source_end_time"]):
                    continue
                if any(other is not shot and shot["identity"] == other["identity"] and
                       max(shot["start"], other["start"]) < min(float(end), other["end"])
                       for other in result["shots"]):
                    continue
                frames = round((end - Fraction(shot["source_start_time"])) * fps)
                added = frames - shot["frames"]
                if added <= 0 or (cap is not None and total_frames + added > cap):
                    continue
                if abs(float(Fraction(cursor + frames, 1) / fps) - beat) > 1 / float(fps) + 1 / float(source_fps):
                    continue
                shot.update({"end": float(end), "source_end_time": str(end),
                             "source_end_frame_exclusive": end_frame,
                             "source_pts_end_exclusive": str((end + origin) / base),
                             "frames": frames, "duration": float(Fraction(frames, 1) / fps),
                             "beat_exit_extension": float(Fraction(added, 1) / fps)})
                result["beat_timing"]["aligned_cuts"] += 1
                extended_frames += added
                total_frames += added
                break
        shot.update({"timeline_in_frame": cursor, "timeline_out_frame": cursor + shot["frames"],
                     "timeline_in": float(Fraction(cursor, 1) / fps)})
        cursor += shot["frames"]
    result.update({"frames": cursor, "duration": float(Fraction(cursor, 1) / fps)})
    result["beat_timing"].update({"extended_seconds": float(Fraction(extended_frames, 1) / fps),
                                  "status": "Safe exits checked against detected music onsets"})
    return result
