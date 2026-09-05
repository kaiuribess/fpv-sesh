"""Source-based social exports: preserve the timeline and expose crop choices."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import uuid

from .media import locate_tools, probe, run, sha256_file, validate_output

SOCIAL_VERSION = "social-source-v2-upload-profile"
PROFILES = {
    "vertical": {"label": "Vertical 9:16", "size": (1080, 1920), "preview": (360, 640),
                 "platforms": ["YouTube Shorts", "Instagram/Facebook Reels", "TikTok"]},
    "square": {"label": "Square 1:1", "size": (1080, 1080), "preview": (360, 360),
               "platforms": ["Square feed video", "YouTube Shorts"]},
    "portrait": {"label": "Portrait 4:5", "size": (1080, 1350), "preview": (360, 450),
                 "platforms": ["Portrait feed video", "YouTube Shorts"]},
}


def social_export_plan(settings, timeline=None, preview=False):
    """Return {format_code: profile} without touching files or changing timing."""
    formats = settings.get("social_formats", [])
    if isinstance(formats, str) or not isinstance(formats, (list, tuple)):
        raise ValueError("social_formats must be a list of vertical, square or portrait")
    framing = settings.get("framing", "blur")
    if framing not in ("blur", "fit", "fill"):
        raise ValueError("framing must be blur, fit or fill")
    focus = settings.get("focus_x", 0.5)
    if isinstance(focus, bool) or not isinstance(focus, (float, int)) or not math.isfinite(focus) or not 0 <= focus <= 1:
        raise ValueError("focus_x must be a finite number from 0 through 1")
    plans = {}
    for code in formats:
        if code not in PROFILES:
            raise ValueError(f"Unknown social format: {code}")
        width, height = PROFILES[code]["preview" if preview else "size"]
        warnings = []
        if framing == "fill":
            warnings.append("Fill crops scene edges. Review tree passes and tricks; horizontal focus is fixed, not subject tracking.")
        if preview:
            warnings.append("This is a small review preview; use the final social export for uploading.")
        if timeline:
            duration = float(timeline.get("duration", 0))
            if duration > 180:
                warnings.append("This timeline exceeds the three-minute YouTube Shorts limit. Its complete duration is preserved.")
                warnings.append("TikTok duration availability depends on the account; this export is not trimmed automatically.")
            if duration < 3 or duration > 900:
                warnings.append("This duration is outside Instagram's documented 3-second to 15-minute publishing API range; the timeline is preserved.")
            fps = Fraction(str(timeline["fps"]))
            if not 23 <= fps <= 60:
                warnings.append("The chosen frame rate is preserved but is outside TikTok's documented 23–60 fps upload range.")
        plans[code] = {
            "code": code, "label": PROFILES[code]["label"], "width": width, "height": height,
            "aspect_ratio": str(Fraction(width, height)), "framing": framing, "focus_x": float(focus),
            "platforms": list(PROFILES[code]["platforms"]), "preview": bool(preview),
            "video_codec": "h264", "pixel_format": "yuv420p", "audio_codec": "aac",
            "audio_sample_rate": 48000, "faststart": True,
            "video_maxrate_bps": 20000000, "audio_bitrate_bps": 128000,
            "duration_policy": "Preserve every selected timeline frame and complete selected maneuver",
            "enhancement": "Original-source Lanczos resizing and selected fixed shot grade",
            "warnings": warnings,
        }
    return plans


def _save_json(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _aspect(metadata):
    try:
        sar = Fraction(metadata.get("sample_aspect_ratio", "1:1").replace(":", "/"))
        if sar <= 0:
            sar = Fraction(1)
    except (ValueError, ZeroDivisionError, AttributeError):
        sar = Fraction(1)
    aspect = Fraction(metadata["width"], metadata["height"]) * sar
    # FFmpeg autorotation happens before the scale/crop graph.
    if round(float(metadata.get("rotation", 0))) % 180 == 90:
        aspect = 1 / aspect
    return aspect


def _dimensions(aspect, width, height, contain):
    if (aspect > Fraction(width, height)) == contain:
        w, h = width, float(Fraction(width, 1) / aspect)
    else:
        w, h = float(Fraction(height, 1) * aspect), height
    rounding = math.floor if contain else math.ceil
    return max(2, rounding(w / 2) * 2), max(2, rounding(h / 2) * 2)


def _graph(metadata, profile, fps, frames, grade):
    w, h = profile["width"], profile["height"]
    aspect = _aspect(metadata)
    fit_w, fit_h = _dimensions(aspect, w, h, True)
    cover_w, cover_h = _dimensions(aspect, w, h, False)
    focus = profile["focus_x"]
    prefix = f"[0:v:0]setpts=PTS-STARTPTS,fps={fps},trim=end_frame={frames},"
    if grade:
        prefix += grade + ","
    scale_opts = ":flags=lanczos:out_color_matrix=bt709:out_range=tv"
    framing = profile["framing"]
    if framing == "fit":
        return (prefix + f"scale={fit_w}:{fit_h}{scale_opts},setsar=1,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p[v]")
    if framing == "fill":
        return (prefix + f"scale={cover_w}:{cover_h}{scale_opts},setsar=1,"
                f"crop={w}:{h}:trunc((iw-ow)*{focus:.9f}/2)*2:trunc((ih-oh)/4)*2,format=yuv420p[v]")
    # Blur a small background and expand it; keep the foreground at full profile
    # resolution. This avoids blurring millions of background pixels per frame.
    bw, bh = max(2, w // 8 * 2), max(2, h // 8 * 2)
    bcw, bch = _dimensions(aspect, bw, bh, False)
    return (prefix + "split=2[foreground][background];"
            f"[foreground]scale={fit_w}:{fit_h}{scale_opts},setsar=1[fg];"
            f"[background]scale={bcw}:{bch}{scale_opts},setsar=1,"
            f"crop={bw}:{bh}:trunc((iw-ow)*{focus:.9f}/2)*2:trunc((ih-oh)/4)*2,"
            f"gblur=sigma=14:steps=2,eq=brightness=-0.08:saturation=0.75,"
            f"scale={w}:{h}:flags=bilinear,setsar=1[bg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p[v]")


def _encoder_args(encoder, fps, preview):
    common = ["-c:v", encoder, "-profile:v", "high", "-pix_fmt", "yuv420p",
              "-g", str(max(1, round(float(Fraction(fps)) * 2))), "-bf", "2"]
    if encoder == "h264_nvenc":
        return common + ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "17",
                         "-b:v", "0", "-maxrate", "20M", "-bufsize", "40M", "-spatial-aq", "1"]
    return common + ["-preset", "fast" if preview else "medium", "-crf", "18" if preview else "17",
                     "-maxrate", "20M", "-bufsize", "40M",
                     "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:open-gop=0"]


def _segment_valid(path, record_path, options):
    if not path.is_file() or not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("options") != options or record.get("_cache_sha256") != sha256_file(path):
            return None
        meta = probe(path, include_hash=False)
        if (meta["frame_count"] != options["frames"] or meta["width"] != options["width"]
                or meta["height"] != options["height"] or Fraction(meta["fps"]) != Fraction(options["fps"])
                or meta["codec"] != "h264" or meta["pix_fmt"] != "yuv420p"):
            return None
        return record
    except (OSError, ValueError, RuntimeError, KeyError):
        return None


def _is_faststart(path):
    positions = {}
    with Path(path).open("rb") as stream:
        length = Path(path).stat().st_size
        while stream.tell() + 8 <= length:
            offset = stream.tell()
            size, atom = struct.unpack(">I4s", stream.read(8))
            header = 8
            if size == 1:
                size = struct.unpack(">Q", stream.read(8))[0]
                header = 16
            elif size == 0:
                size = length - offset
            if size < header or offset + size > length:
                return False
            positions.setdefault(atom, offset)
            if b"moov" in positions and b"mdat" in positions:
                return positions[b"moov"] < positions[b"mdat"]
            stream.seek(offset + size)
    return False


def _audio_args(ffprobe, job, duration):
    audio = job / "source-audio.m4a"
    if not audio.is_file():
        return (["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"],
                ["-map", "1:a:0", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"], "silence")
    result = run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_streams", "-of", "json", str(audio)])
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("Canonical source-audio.m4a contains no audio stream")
    stream = streams[0]
    copy_ok = (stream.get("codec_name") == "aac" and str(stream.get("sample_rate")) == "48000"
               and stream.get("channels") == 2 and int(stream.get("bit_rate", 999999)) <= 128000
               and abs(float(stream.get("duration", 0)) - duration) <= .02)
    if copy_ok:
        return ["-i", str(audio)], ["-map", "1:a:0", "-c:a", "copy"], "canonical AAC copied"
    return (["-i", str(audio)], ["-map", "1:a:0", "-af", f"aresample=48000,apad,atrim=duration={duration:.9f},asetpts=N/SR/TB",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"], "canonical audio normalized to AAC48kHz128kbps")


def export_social(timeline, probes, settings, job, cache, event, checkpoint, preview=False):
    """Render selected social formats directly from original source intervals.

    Returns ``{code: {path, verification, records, profile, poster, warnings}}``.
    A checkpoint is consulted before/after each segment and before verification;
    completed cache files are authenticated before reuse. No remote publishing.
    """
    plans = social_export_plan(settings, timeline, preview)
    if not plans:
        return {}
    from .render import bound_cache, ensure_space, grade_for
    shots = timeline.get("shots", [])
    fps_fraction = Fraction(str(timeline["fps"]))
    if not shots or fps_fraction <= 0 or any(int(s.get("frames", 0)) <= 0 for s in shots):
        raise ValueError("Social exports require a nonempty frame-counted timeline")
    fps = f"{fps_fraction.numerator}/{fps_fraction.denominator}"
    total = sum(int(s["frames"]) for s in shots)
    if timeline.get("frames", total) != total:
        raise ValueError("Timeline total differs from its selected shot frame counts")
    duration = float(Fraction(total, 1) / fps_fraction)
    job, cache = Path(job).resolve(), Path(cache).resolve()
    output_dir = job / ("social-preview" if preview else "social")
    segment_dir = cache / "segments"
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir.mkdir(parents=True, exist_ok=True)
    ensure_space(cache, max(128 * 2**20, int(duration * len(plans) * (1_000_000 if preview else 6_000_000))))
    ffmpeg, ffprobe = locate_tools()
    ffmpeg_stat = Path(ffmpeg).stat()
    by_source = {os.path.normcase(str(Path(p["source"]).resolve())): p for p in probes}
    source_contracts = {}
    for shot in shots:
        source = Path(shot["source"]).resolve(strict=True)
        key = os.path.normcase(str(source))
        if key not in by_source:
            raise ValueError(f"Missing source probe: {source.name}")
        metadata = by_source[key]
        if metadata.get("hdr"):
            raise ValueError("HDR source requires an explicit SDR conversion before these social profiles")
        stat = source.stat()
        if metadata.get("size_bytes", stat.st_size) != stat.st_size or metadata.get("mtime_ns", stat.st_mtime_ns) != stat.st_mtime_ns:
            raise ValueError(f"Original source changed after probing: {source.name}")
        if key not in source_contracts:
            identity = metadata.get("identity") or shot.get("identity") or {}
            if not isinstance(identity, dict) or not identity.get("sha256"):
                identity = {"sha256": sha256_file(source), "size_bytes": stat.st_size}
            source_contracts[key] = (source, metadata, identity, stat.st_size, stat.st_mtime_ns)
    source_paths = {item[0] for item in source_contracts.values()}
    for code in plans:
        if (output_dir / f"{code}.mp4").resolve() in source_paths:
            raise ValueError("Social output would overwrite an original source")
    results = {}
    progress_done = 0
    stage = "social-preview" if preview else "social"
    for code, profile in plans.items():
        paths, records = [], []
        for index, shot in enumerate(shots):
            checkpoint()
            source, metadata, identity, size, mtime = source_contracts[os.path.normcase(str(Path(shot["source"]).resolve()))]
            start = float(Fraction(str(shot.get("source_start_time", shot["start"]))))
            if not math.isfinite(start) or start < 0:
                raise ValueError("Source start must be a finite nonnegative time")
            grade = grade_for(shot, settings.get("look", "natural"), settings.get("strength", 0))
            frames = int(shot["frames"])
            options = {
                "version": SOCIAL_VERSION, "identity": identity, "source": str(source), "source_mtime_ns": mtime,
                "start": start, "source_start_frame": shot.get("source_start_frame"),
                "source_end_frame_exclusive": shot.get("source_end_frame_exclusive"), "frames": frames, "fps": fps,
                "width": profile["width"], "height": profile["height"], "profile": code,
                "framing": profile["framing"], "focus_x": profile["focus_x"], "grade": grade,
                "rotation": metadata.get("rotation", 0), "sar": metadata.get("sample_aspect_ratio"),
                "preview": bool(preview), "encoder_policy": "libx264" if preview else "h264_nvenc_then_libx264",
                "tool": {"path": ffmpeg, "size": ffmpeg_stat.st_size, "mtime_ns": ffmpeg_stat.st_mtime_ns},
            }
            digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:24]
            target = segment_dir / f"{digest}.mp4"
            record_path = target.with_suffix(".json")
            if target.resolve() in source_paths:
                raise ValueError("Segment output would overwrite an original source")
            event(stage, progress_done / (total * len(plans)), f"{profile['label']}: shot {index+1}/{len(shots)}")
            bound_cache(cache, reserve=64 * 2**20, protected=paths)
            record = _segment_valid(target, record_path, options)
            if record is None:
                temporary = target.with_name(target.stem + "." + uuid.uuid4().hex + ".partial.mp4")
                try:
                    base = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{start:.9f}",
                            "-i", str(source), "-filter_complex", _graph(metadata, profile, fps, frames, grade),
                            "-map", "[v]", "-an", "-frames:v", str(frames), "-r", fps,
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
                    encoders = ["libx264"] if preview else ["h264_nvenc", "libx264"]
                    failures = []
                    for encoder in encoders:
                        completed = run(base + _encoder_args(encoder, fps, preview) + [str(temporary)],
                                        log_file=job / "social-render.log", check=False)
                        if completed.returncode == 0:
                            break
                        failures.append({"encoder": encoder, "error": completed.stderr[-1600:]})
                        temporary.unlink(missing_ok=True)
                        event(stage, progress_done / (total * len(plans)), f"{encoder} unavailable; trying the next social encoder")
                    else:
                        raise RuntimeError(f"All social encoders failed: {failures}")
                    meta = probe(temporary, include_hash=False)
                    if (meta["frame_count"] != frames or (meta["width"], meta["height"]) != (profile["width"], profile["height"])
                            or Fraction(meta["fps"]) != fps_fraction or meta["codec"] != "h264" or meta["pix_fmt"] != "yuv420p"):
                        raise RuntimeError("Social segment failed its frame, format or dimension checkpoint")
                    if (source.stat().st_size, source.stat().st_mtime_ns) != (size, mtime):
                        raise RuntimeError("Original source changed while the segment was rendering")
                    checkpoint()
                    temporary.replace(target)
                    record = {"options": options, "encoder": encoder, "gpu_inference": False,
                              "backend": "Original-source Lanczos social composition", "failures": failures,
                              "source": str(source), "source_start": start, "frames": frames,
                              "output_path": str(target), "_cache_sha256": sha256_file(target)}
                    _save_json(record_path, record)
                finally:
                    temporary.unlink(missing_ok=True)
            paths.append(target)
            records.append(record)
            progress_done += frames
            _save_json(output_dir / f"{code}-checkpoint.json", {"completed_frames": sum(r["frames"] for r in records),
                        "total_frames": total, "segments": [str(p) for p in paths]})
        checkpoint()
        target = output_dir / f"{code}.mp4"
        token = uuid.uuid4().hex
        temporary = output_dir / f"{code}.{token}.partial.mp4"
        listing = segment_dir / f"social-{token}.txt"
        listing.write_text("\n".join(f"file '{p.name}'" for p in paths) + "\n", encoding="utf-8")
        try:
            inputs, audio_options, audio_mode = _audio_args(ffprobe, job, duration)
            run([ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "1", "-i", str(listing)]
                + inputs + ["-map", "0:v:0", "-c:v", "copy"] + audio_options
                + ["-t", f"{duration:.9f}", "-movflags", "+faststart", "-video_track_timescale", str(fps_fraction.numerator), str(temporary)],
                log_file=job / "social-render.log")
            checkpoint()
            event(stage, progress_done / (total * len(plans)), f"Verifying every {profile['label']} frame")
            verification = validate_output(temporary, total, fps, profile["width"], profile["height"], log_file=job / "social-render.log")
            meta = verification["probe"]
            extra_errors = []
            if meta["codec"] != "h264" or meta["pix_fmt"] != "yuv420p":
                extra_errors.append("Social output must be H264 8-bit 4:2:0")
            if not meta["audio_streams"] or any(a.get("codec_name") != "aac" or str(a.get("sample_rate")) != "48000" for a in meta["audio_streams"]):
                extra_errors.append("Social output must contain AAC48kHz audio")
            faststart = _is_faststart(temporary)
            if not faststart:
                extra_errors.append("Social MP4 does not have its index before media data")
            verification["errors"].extend(extra_errors)
            verification["passed"] = not verification["errors"]
            verification["social"] = {"profile": code, "framing": profile["framing"], "audio_mode": audio_mode, "faststart": faststart}
            if not verification["passed"]:
                _save_json(output_dir / f"{code}-verification.json", verification)
                for path in paths:
                    path.with_suffix(".json").unlink(missing_ok=True)
                raise RuntimeError(f"Social output verification failed: {verification['errors']}")
            checkpoint()
            temporary.replace(target)
            verification["probe"]["source"] = str(target)
            verification["probe"]["raw"]["format"]["filename"] = str(target)
            poster = output_dir / f"{code}-poster.jpg"
            # The first selected shot is the chosen opener; use a frame within it.
            poster_time = min(float(Fraction(shots[0]["frames"] - 1, 1) / fps_fraction), 0.5)
            poster_tmp = poster.with_name(f"{code}.{token}.poster.jpg")
            try:
                photo = run([ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-ss", str(poster_time),
                             "-i", str(target), "-frames:v", "1", "-q:v", "2", str(poster_tmp)], check=False)
                if photo.returncode == 0 and poster_tmp.is_file():
                    poster_tmp.replace(poster)
                    poster_path = str(poster)
                else:
                    poster_path = None
            finally:
                poster_tmp.unlink(missing_ok=True)
            result = {"path": str(target), "verification": verification, "records": records,
                      "profile": profile, "poster": poster_path, "warnings": list(profile["warnings"])}
            _save_json(output_dir / f"{code}-verification.json", verification)
            _save_json(output_dir / f"{code}-backends.json", records)
            results[code] = result
        finally:
            temporary.unlink(missing_ok=True)
            listing.unlink(missing_ok=True)
    event(stage, 1, "All requested social exports verified")
    return results
