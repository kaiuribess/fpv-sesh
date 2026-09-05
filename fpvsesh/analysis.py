"""Full-session proxy analysis. Motion/semantic descriptions are estimates."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector
from .media import locate_tools, run

ANALYSIS_VERSION = "5"
PROXY_FPS = 12
DETAIL_VERSION = 1

def terminal_arrival(rows, duration):
    """Estimate terminal settling from persistent appearance AND low motion.

    This is not a crash classifier. A terminal hover can also match, so the
    estimate remains reviewable and explicit keep overrides are permitted.
    """
    if len(rows) < 24 or duration < 6:
        return None
    tail = [r for r in rows if r["t"] >= duration - 2.3]
    if len(tail) < 10 or not all("hash" in r for r in tail): return None
    reference = int(tail[-1]["hash"], 16)
    if (np.median([r["motion"] for r in tail]) >= .4 or
        np.percentile([r["motion"] for r in tail], 90) >= .8 or
        np.median([(int(r["hash"],16)^reference).bit_count() for r in tail]) > 5):
        return None
    index = max(0, len(rows)-len(tail))
    while index > 0:
        window = rows[index-1:min(len(rows),index+11)]
        if (np.percentile([r["motion"] for r in window], 80) > .8 or
            np.median([(int(r.get("hash","0"),16)^reference).bit_count() for r in window]) > 6): break
        index -= 1
    settled = float(rows[index]["t"])
    prior = [r for r in rows if max(0,settled-20) <= r["t"] < settled]
    if not prior or max(r["motion"] for r in prior) < 2:
        return None
    return {"settled_start": settled, "exclude_start": max(0,settled-10), "confidence": .72,
            "evidence": "persistent terminal framing and low motion after activity; 10-second arrival context excluded automatically, not a crash diagnosis"}

def _enrich_flight_detail(result, event, checkpoint):
    if result.get("detail_version") == DETAIL_VERSION: return result
    cap = cv2.VideoCapture(result["proxy"])
    previous, idx, count = None, 0, 0
    grid_y, grid_x = np.mgrid[0:192,0:256].astype(np.float32)
    xy = np.stack([grid_x,grid_y,np.ones_like(grid_x)],axis=-1)
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % 2:
            idx += 1
            continue
        thumb = cv2.resize(frame,(256,192),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumb,cv2.COLOR_BGR2GRAY)
        detail = {"residual_motion":0.0,"proximity":0.0,"parallax_confidence":0.0,"foreground_texture":0.0}
        if previous is not None:
            points = cv2.goodFeaturesToTrack(previous,maxCorners=180,qualityLevel=.012,minDistance=6)
            if points is not None and len(points) >= 20:
                dest,status,_ = cv2.calcOpticalFlowPyrLK(previous,gray,points,None)
                if dest is not None and status.sum() >= 20:
                    good = status.ravel() == 1
                    affine,inliers = cv2.estimateAffinePartial2D(points[good],dest[good],method=cv2.RANSAC,ransacReprojThreshold=2.5)
                    if affine is not None and inliers is not None:
                        confidence = float(np.mean(inliers))
                        flow = cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,19,3,5,1.2,0)
                        predicted = xy @ affine.T - xy[:,:,:2]
                        residual = np.linalg.norm(flow-predicted,axis=2)
                        hsv = cv2.cvtColor(thumb,cv2.COLOR_BGR2HSV)
                        edge = np.abs(cv2.Laplacian(gray,cv2.CV_32F))
                        # A textured, colored foreground is supporting evidence;
                        # grass/structures may match too. This is not tree recognition.
                        foreground = (hsv[:,:,0] >= 13) & (hsv[:,:,0] <= 100) & (hsv[:,:,1] > 40) & (hsv[:,:,2] > 20) & (edge > 9)
                        region = np.zeros_like(foreground);region[35:153,8:248] = True
                        foreground &= region
                        coverage = float(foreground.sum()/max(1,region.sum()))
                        residual_value = float(np.percentile(residual[foreground],85)) if foreground.sum() >= 150 else 0
                        confidence = max(0,min(1,(confidence-.25)/.5))
                        detail = {"residual_motion":round(residual_value,4),
                                  "proximity":round(min(1,residual_value/7)*min(1,coverage/.15),4),
                                  "parallax_confidence":round(confidence,4),"foreground_texture":round(coverage,4)}
        if count >= len(result["rows"]): break
        result["rows"][count].update(detail)
        previous = gray;count += 1
        if idx % 240 == 0:
            event("analysis",idx/12/max(result["duration"],1),f"Checking close passes and landing context: {Path(result['source']).name} {idx/12:.0f}s")
            checkpoint()
        idx += 1
    cap.release()
    if count != len(result["rows"]): raise RuntimeError("Detailed proxy analysis did not cover the complete session")
    result["detail_version"] = DETAIL_VERSION
    result["terminal_arrival"] = terminal_arrival(result["rows"],result["duration"])
    return result

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def identity(p):
    return p.get("sha256") or p.get("identity") or hashlib.sha256(p["source"].encode()).hexdigest()

def analyze(p, cache: Path, event, checkpoint=lambda: None):
    key = hashlib.sha256((str(identity(p)) + ANALYSIS_VERSION).encode()).hexdigest()[:20]
    folder = cache / "analysis" / key
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "analysis.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["source"] = p["source"]
        result = _enrich_flight_detail(result,event,checkpoint)
        save_json(result_path,result)
        event("analysis", 1, f"Reusing full-session analysis: {Path(p['source']).name}")
        return result
    proxy = folder / "proxy.mp4"
    ffmpeg, _ = locate_tools()
    if not proxy.exists():
        event("proxy", 0, f"Making full-length proxy: {Path(p['source']).name}")
        temp = folder / "proxy.partial.mp4"
        run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", p["source"], "-map", "0:v:0", "-an",
             "-vf", "fps=12,scale=480:360:force_original_aspect_ratio=decrease,setsar=1", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", str(temp)], log_file=folder / "proxy.log")
        temp.replace(proxy)
    checkpoint()
    # Adaptive cuts are evidence only. Fast flight can also cause abrupt changes.
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector(adaptive_threshold=5, min_scene_len=24, min_content_val=40))
    video = open_video(str(proxy))
    manager.detect_scenes(video=video, show_progress=False)
    cuts = [a.get_seconds() for a, _ in manager.get_scene_list() if a.get_seconds() > 0]
    cap = cv2.VideoCapture(str(proxy))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, previous, idx = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 2:
            idx += 1
            continue
        thumb = cv2.resize(frame, (256, 192), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        # Exclude outer corners (props/lens borders) from quality statistics.
        center = gray[14:-14, 20:-20]
        hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
        motion = rotation = dx = dy = 0.0
        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 19, 3, 5, 1.2, 0)
            inner = flow[14:-14, 20:-20]
            motion = float(np.percentile(np.linalg.norm(inner, axis=2), 65))
            dx, dy = map(float, np.median(inner.reshape(-1, 2), axis=0))
            pts = cv2.goodFeaturesToTrack(previous, maxCorners=100, qualityLevel=.015, minDistance=9)
            if pts is not None and len(pts) >= 10:
                dest, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, pts, None)
                if dest is not None and status.sum() >= 8:
                    good = status.ravel() == 1
                    affine, _ = cv2.estimateAffinePartial2D(pts[good], dest[good], method=cv2.RANSAC)
                    if affine is not None:
                        rotation = float(math.degrees(math.atan2(affine[1, 0], affine[0, 0])))
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
        hist /= max(float(hist.sum()), 1)
        tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = (tiny[:, 1:] > tiny[:, :-1]).ravel()
        phash = hex(sum(int(bit) << j for j, bit in enumerate(bits)))
        rows.append({"t": idx / PROXY_FPS, "motion": round(motion, 4), "rotation": round(rotation, 3),
                     "dx": round(dx, 3), "dy": round(dy, 3), "sharpness": round(float(cv2.Laplacian(center, cv2.CV_32F).var()), 2),
                     "luma": round(float(center.mean()) / 255, 4), "contrast": round(float(center.std()) / 255, 4),
                     "black": round(float(np.mean(center < 12)), 4), "white": round(float(np.mean(center > 246)), 4),
                     "hist": [round(float(x), 4) for x in hist], "hash": phash})
        previous = gray
        if idx % 240 == 0:
            event("analysis", idx / max(frames, 1), f"Analyzing {Path(p['source']).name}: {idx / 12:.0f}/{p['duration']:.0f} s")
            checkpoint()
        idx += 1
    cap.release()
    if len(rows) < 3 or idx < frames - 3:
        raise RuntimeError(f"Proxy decode incomplete for {p['source']}")
    result = {"version": ANALYSIS_VERSION, "source": p["source"], "identity": identity(p), "proxy": str(proxy.resolve()),
              "duration": p["duration"], "sample_fps": 6, "coverage_seconds": idx / 12,
              "cuts_estimated": cuts, "rows": rows, "semantic_confidence": "heuristic estimates; no trained trick classifier"}
    result = _enrich_flight_detail(result,event,checkpoint)
    save_json(result_path, result)
    return result

def guard_recovery_end(rows, start, end, duration, hold=2.5):
    """Keep follow-through after estimated motion bursts, including linked ones.

    A quiet individual sample can be the pause between flips. Look back across
    the entire hold, then restart it for any later burst included by extension.
    Rotation and vertical flow are estimates, not proof of an upright recovery
    or a maneuver classifier. None means the source ends before the hold fits.
    Explicitly reviewed source boundaries bypass this automatic policy.
    """
    if hold <= 0:
        raise ValueError("Recovery hold must be positive")
    protected_end = float(end)
    for row in rows:
        time = row["t"]
        if time < max(start, end - hold):
            continue
        if time > protected_end or time >= duration:
            break
        burst = (abs(row.get("rotation", 0)) >= 8 or
                 (abs(row.get("dy", 0)) >= 10 and row.get("motion", 0) >= 8))
        if burst:
            protected_end = max(protected_end, time + hold)
            if protected_end > duration + 1e-6:
                return None
    return protected_end if protected_end <= duration + 1e-6 else None


def candidates_from_analysis(analyses, style="hype", reviewed_intervals=None):
    candidates = []
    reviewed_intervals = reviewed_intervals or []
    sources = {a["source"]:a for a in analyses}
    for reviewed in reviewed_intervals:
        if reviewed["source"] not in sources or not (0 <= reviewed["start"] < reviewed["end"] <= sources[reviewed["source"]]["duration"]):
            raise ValueError("Reviewed interval is outside the analyzed source bounds")
    for a in analyses:
        rows = a["rows"]
        arrival = a.get("terminal_arrival") or terminal_arrival(rows,a["duration"])
        t = np.array([r["t"] for r in rows])
        motion = np.array([r["motion"] for r in rows])
        smooth = np.convolve(motion, np.ones(7) / 7, mode="same")
        rotations = np.abs([r["rotation"] for r in rows])
        scale = max(float(np.percentile(motion, 85)), 1)
        length = min({"hype": 7.2, "cinematic": 10.0, "freestyle": 9.5}[style], a["duration"]-.4)
        centers = list(np.arange(length / 2 + .2, a["duration"] - 2, 3.5))
        if not centers and a["duration"] >= 3.4:
            centers = [a["duration"] / 2]
        # Extra action peaks are considered over the entire session.
        peaks = np.where((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:]) & (smooth[1:-1] > scale))[0] + 1
        centers.extend(float(t[x]) for x in peaks if 3 < t[x] < a["duration"] - 3)
        seen = set()
        windows = [((r["start"]+r["end"])/2,r) for r in reviewed_intervals if r["source"] == a["source"]]
        windows += [(center,None) for center in sorted(centers)]
        for center, reviewed in windows:
            start = reviewed["start"] if reviewed else max(.1, center - length / 2)
            end = reviewed["end"] if reviewed else min(a["duration"] - .1, center + length / 2)
            # Choose nearby calmer boundaries; rotation itself is never a rejection reason.
            for boundary in (() if reviewed else ("start", "end")):
                point = start if boundary == "start" else end
                eligible = np.flatnonzero((t >= point - .8) & (t <= point + .8) & (t >= .1) & (t < a["duration"] - .1))
                if len(eligible):
                    costs = rotations[eligible] / 3 + np.abs(t[eligible] - point) * .8
                    best = float(t[eligible[np.argmin(costs)]])
                    if boundary == "start": start = best
                    else: end = best
            # Retain approach when the opening boundary is inside a rotation.
            while not reviewed and start > .7 and rotations[min(len(t) - 1, int(start * 6))] > 8 and center - start < 6:
                start -= .5
            recovery_incomplete = False
            if not reviewed:
                guarded_end = guard_recovery_end(rows, start, end, a["duration"] - .1)
                recovery_incomplete = guarded_end is None
                if guarded_end is not None:
                    end = guarded_end
            if end - start < 3 or (round(start), round(end)) in seen:
                continue
            seen.add((round(start), round(end)))
            chunk = [r for r in rows if start <= r["t"] < end]
            m = np.array([r["motion"] for r in chunk])
            idle = float(np.mean(m < .22))
            mean = lambda k: float(np.mean([r[k] for r in chunk]))
            activity = min(float(np.percentile(m, 70)) / scale, 1.5)
            sharp = min(math.log1p(mean("sharpness")) / 6, 1)
            exposure = max(0, 1 - mean("black") * 1.8 - max(0, mean("white") - .12) * 2)
            turn = min(float(np.percentile([abs(r["rotation"]) for r in chunk], 80)) / 10, 1)
            activity_weight, turn_weight = {"hype":(18,8), "cinematic":(18,4), "freestyle":(20,20)}[style]
            proximity = float(np.mean([r.get("proximity",0)*r.get("parallax_confidence",0) for r in chunk]))
            score = activity_weight * activity + 20 * sharp + 20 * exposure + 10 * min(mean("contrast") / .2, 1) + turn_weight * turn + 30*proximity - 65 * idle
            if style == "cinematic": score += 8 * (1-turn)
            unusable = idle > .7 or mean("black") > .78 or mean("contrast") < .015 or recovery_incomplete
            reason = "flowing line / pass estimate; approach and exit context retained"
            if turn > .5: reason = "turn or maneuver estimate; rotation preserved with context"
            if unusable: reason = "prolonged idle / obscured or unusable image"
            elif idle > .35: reason = "possible settling / reveal; partially idle"
            end_rotation = float(np.mean([abs(r["rotation"]) for r in chunk[-6:]]))
            after = [r for r in rows if end <= r["t"] < end + 5]
            following_idle = float(np.mean([r["motion"] < .22 for r in after])) if after else 0
            if following_idle > .4 and end_rotation > 5:
                score -= 18
                reason += "; unsettled exit before stationary tail, down-ranked (not a crash classification)"
            arrival_estimate = bool(arrival and end > arrival["exclude_start"])
            if arrival_estimate:
                unusable = True
                reason = "terminal arrival / settled-ground context estimate; excluded from automatic highlights, explicit keep remains available"
            elif proximity > .28:
                reason = "close structure / weaving estimate from residual foreground motion; approach and exit retained"
            if reviewed:
                reason = reviewed["reason"] + "; editorial review from sampled source frames"
            elif recovery_incomplete:
                reason = "recording ends before estimated maneuver follow-through fits; excluded automatically, explicit keep remains available"
            elif not arrival_estimate:
                reason += "; at least 2.5 seconds retained after the last detected motion burst"
            seq = [chunk[int((len(chunk)-1)*f)]["hash"] for f in (.2, .5, .8)]
            cid = hashlib.sha256(f"{a['identity']}:{start:.4f}:{end:.4f}".encode()).hexdigest()[:12]
            candidates.append({"id": cid, "source": a["source"], "identity": a["identity"], "start": round(start, 6), "end": round(end, 6),
                               "score": round(score, 3), "confidence": reviewed.get("confidence",.8) if reviewed else .58, "reason": reason, "selected": False, "unusable": unusable,
                               "review_key": reviewed.get("key") if reviewed else None,
                               "recovery_incomplete": recovery_incomplete,
                               "recovery_hold_seconds": None if reviewed else 2.5,
                               "motion": round(mean("motion"), 4), "rotation": round(turn, 4), "idle": round(idle, 4), "luma": mean("luma"),
                               "dx_in": float(np.mean([r["dx"] for r in chunk[:6]])), "dx_out": float(np.mean([r["dx"] for r in chunk[-6:]])),
                               "end_rotation": end_rotation, "following_idle": following_idle,
                               "proximity": proximity, "arrival_estimate": arrival_estimate,
                               "hist": np.mean([r["hist"] for r in chunk], axis=0).tolist(), "hash_sequence": seq,
                               "source_duration": a["duration"], "cut_evidence": [c for c in a["cuts_estimated"] if start < c < end]})
    return candidates
