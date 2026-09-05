from __future__ import annotations
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import shutil
import re
from .media import locate_tools, run, probe, validate_output, sha256_file
from .analysis import save_json

RENDER_VERSION = "5-detail-restoration-encode-headroom"

def ensure_space(path, needed_bytes):
    free = shutil.disk_usage(path).free
    if free < needed_bytes:
        raise RuntimeError(f"Not enough free disk space: need {needed_bytes / 2**30:.1f} GiB, available {free / 2**30:.1f} GiB")

def bound_cache(cache, reserve=0, protected=(), limit=40 * 2**30):
    """Evict only completed application segment cache entries, never arbitrary paths."""
    cache = Path(cache).resolve()
    files = [p for p in cache.rglob("*") if p.is_file() and not p.is_symlink()]
    total = sum(p.stat().st_size for p in files)
    protected = {Path(p).resolve() for p in protected}
    segment_dir = cache / "segments"
    evictable = sorted((p for p in files if p.parent == segment_dir and re.fullmatch(r"[0-9a-f]{24}\.mp4",p.name) and p.resolve() not in protected), key=lambda p:p.stat().st_mtime)
    for path in evictable:
        if total + reserve <= limit: break
        for owned in (path,path.with_suffix(".json")):
            if owned.exists() and owned.resolve().parent == segment_dir:
                total -= owned.stat().st_size
                owned.unlink()
    if total + reserve > limit:
        raise RuntimeError("The 40 GiB application cache budget cannot hold this job. Remove old job cache through the documented cache cleanup procedure or use conventional enhancement.")

def grade_for(shot, look, strength):
    # Fixed per-shot adjustments: never frame-by-frame pumping or sunset neutralization.
    strength = max(0, min(1, strength))
    if strength == 0:
        return ""
    gamma = 1 + max(-.015, min(.025, (.36 - shot.get("luma", .36)) * .12)) * strength
    contrast = 1 + {"punch": .035, "natural": 0, "cinematic": -.015}[look] * strength
    saturation = 1 + {"punch": .045, "natural": 0, "cinematic": -.06}[look] * strength
    return f"eq=gamma={gamma:.5f}:contrast={contrast:.5f}:saturation={saturation:.5f}"

def _concat_file(paths, dest):
    # Generated app-owned segments have simple relative names, avoiding concat escaping of source names.
    dest.write_text("\n".join("file '" + p.name + "'" for p in paths) + "\n", encoding="utf-8")

def make_audio(timeline, probes, job, level, event, checkpoint):
    ffmpeg, _ = locate_tools()
    by_source = {p["source"]: p for p in probes}
    if level <= 0 or not any(by_source[s["source"]]["audio"] for s in timeline["shots"]): return None
    directory = job / "audio"
    directory.mkdir(exist_ok=True)
    paths = []
    for i, s in enumerate(timeline["shots"]):
        checkpoint()
        out = directory / f"shot-{i:03d}.wav"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        if by_source[s["source"]]["audio"]:
            command += ["-ss", f"{s['start']:.9f}", "-i", s["source"], "-map", "0:a:0"]
        else:
            command += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_filter = f"aresample=48000:async=1:first_pts=0,apad,atrim=duration={s['duration']:.9f},asetpts=N/SR/TB,afade=t=in:d=0.015,afade=t=out:st={max(0,s['duration']-.025):.9f}:d=0.025"
        command += ["-af", audio_filter, "-t", f"{s['duration']:.9f}", "-ac", "2", "-c:a", "pcm_s16le", str(out)]
        run(command, log_file=job / "render.log")
        paths.append(out)
    listing = directory / "list.txt"
    _concat_file(paths, listing)
    joined = directory / "joined.wav"
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)], log_file=job / "render.log")
    # Keep original audio natural; limit peaks and fade the ending, without loudness pumping.
    output = job / "source-audio.m4a"
    audio_filter = f"volume={level:.4f},alimiter=limit=0.8913:level=false:latency=true,afade=t=out:st={max(0,timeline['duration']-.3):.9f}:d=0.3"
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(joined), "-af", audio_filter, "-c:a", "aac", "-b:a", "192k", "-t", f"{timeline['duration']:.9f}", str(output)], log_file=job / "render.log")
    measure = run([ffmpeg, "-hide_banner", "-i", str(output), "-af", "ebur128=peak=true", "-f", "null", "-"], check=False)
    (job / "audio-loudness.txt").write_text(measure.stderr, encoding="utf-8")
    event("audio", 1, "Original audio edited and faded; no music added")
    return output

def render_timeline(timeline, probes, settings, job, cache, event, checkpoint, preview=False):
    ffmpeg, _ = locate_tools()
    stage = "preview" if preview else "final"
    width, height = (1280, 720) if preview else (3840, 2160)
    ensure_space(cache, 2**30 if preview else max(5 * 2**30, timeline["duration"] * 120_000_000 if settings["quality"] == "ai" else 5 * 2**30))
    directory = cache / "segments"
    directory.mkdir(exist_ok=True)
    segments, records = [], []
    total = sum(s["frames"] for s in timeline["shots"])
    done = 0
    for i, s in enumerate(timeline["shots"]):
        checkpoint()
        bound_cache(cache, reserve=512 * 2**20, protected=segments)
        grade = grade_for(s, settings["look"], settings["strength"])
        options = {"version": RENDER_VERSION, "identity": s["identity"], "start": s["start"], "duration": s["duration"],
                   "frames": s["frames"], "fps": timeline["fps"], "vfr": s.get("vfr", False), "width": width, "height": height, "grade": grade,
                   "quality": "lanczos" if preview else settings["quality"], "codec": "h264" if preview else settings["codec"]}
        if not preview and settings["quality"] == "ai":
            from .cuda_backend import signature
            options["ai_signature"] = signature()
        key = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:24]
        out = directory / f"{key}.mp4"
        record_path = out.with_suffix(".json")
        event(stage, done / total, f"{stage.title()} shot {i+1}/{len(timeline['shots'])}: {Path(s['source']).name} {s['start']:.2f}–{s['end']:.2f}s")
        record = None
        if out.exists() and record_path.exists():
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if not isinstance(record, dict) or record.get("_cache_sha256") != sha256_file(out):
                    record = None
                else:
                    meta = probe(out, include_hash=False)
                    if (meta.get("frame_count") != s["frames"]
                            or (meta["width"],meta["height"]) != (width,height)
                            or Fraction(meta["fps"]) != Fraction(timeline["fps"])):
                        record = None
            except (ValueError, RuntimeError, OSError): record = None
        if out.exists() and record is None: out.unlink()
        if not out.exists():
            tmp = out.with_name(out.stem + ".partial.mp4")
            if preview:
                vf = f"setpts=PTS-STARTPTS,fps={timeline['fps']},scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1," + (grade + "," if grade else "") + "format=yuv420p"
                run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{s['start']:.9f}", "-i", s["source"], "-an", "-vf", vf,
                     "-frames:v", str(s["frames"]), "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709", "-r", timeline["fps"], "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", str(tmp)], log_file=job / "render.log")
                record = {"backend": "CPU Lanczos preview", "gpu_inference": False, "encoder": "libx264"}
            else:
                from .enhance import enhance_segment
                record = enhance_segment(Path(s["source"]), tmp, options, lambda message: event(stage, done / total, str(message)))
            meta = probe(tmp, include_hash=False)
            if meta["frame_count"] != s["frames"] or (meta["width"], meta["height"]) != (width, height):
                raise RuntimeError(f"Rendered segment failed frame/dimension verification: {meta['frame_count']} vs {s['frames']}")
            tmp.replace(out)
            tmp.with_suffix(tmp.suffix + ".enhance.json").unlink(missing_ok=True)
            record["output_path"] = str(out)
            # Metadata alone can survive torn/corrupt video packets. Every
            # completed checkpoint authenticates all bytes before future reuse.
            record["_cache_sha256"] = sha256_file(out)
            save_json(record_path, record)
        segments.append(out)
        records.append(record)
        done += s["frames"]
        save_json(job / f"{stage}-checkpoint.json", {"completed_frames": done, "total_frames": total, "segments": [str(x) for x in segments]})
    checkpoint()
    # All segments are the final encode for their selected source interval. Joining does not recompress video.
    listing = directory / f"{job.name}-{stage}-list.txt"
    _concat_file(segments, listing)
    audio_path = job / "source-audio.m4a"
    target = job / ("preview.mp4" if preview else "final_4k.mp4")
    temporary = target.with_name(target.stem + ".partial.mp4")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if audio_path.exists(): command += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else: command += ["-map", "0:v:0", "-an"]
    command += ["-c:v", "copy", "-movflags", "+faststart", "-video_track_timescale", str(Fraction(timeline["fps"]).numerator), str(temporary)]
    run(command, log_file=job / "render.log")
    event("verify", 0, f"Checking every decoded frame and timestamp in the {stage}")
    verification = validate_output(temporary, total, timeline["fps"], width, height)
    if not verification["passed"]:
        save_json(job / f"{stage}-verification.json", verification)
        # New encode/concatenation defects must not poison repeated attempts.
        # Keep outputs for diagnosis, but revoke these application-owned cache
        # records so the next run regenerates the assembled video segments.
        for segment in segments:
            segment.with_suffix(".json").unlink(missing_ok=True)
        raise RuntimeError(f"Output verification failed: {verification['errors']}")
    temporary.replace(target)
    verification["probe"]["source"] = str(target)
    verification["probe"]["raw"]["format"]["filename"] = str(target)
    save_json(job / f"{stage}-verification.json", verification)
    save_json(job / f"{stage}-backends.json", records)
    event(stage, 1, f"Verified {target.name}")
    return {"path": str(target), "records": records, "verification": verification}
