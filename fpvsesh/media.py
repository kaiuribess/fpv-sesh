"""Local, non-destructive media inspection and verification.

All subprocesses receive argument lists with ``shell=False``.  Source identities
are complete SHA-256 hashes, never samples. Frame rates and timestamp arithmetic
use rationals; floating-point seconds are convenience values for reports only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
from collections import Counter
from fractions import Fraction
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


def run(cmd: Sequence[str | os.PathLike[str]], log_file=None, timeout=None,
        *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Execute without a shell; optionally append command/output to a UTF-8 log."""
    if isinstance(cmd, (str, bytes)):
        raise TypeError("Commands must be argument lists, never shell strings.")
    args = [os.fspath(arg) for arg in cmd]
    result = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(args, ensure_ascii=False) + "\n")
            stream.write(result.stdout + result.stderr)
            stream.write(f"\nExit code: {result.returncode}\n")
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()[-6000:]
        raise RuntimeError(f"{Path(args[0]).name} failed ({result.returncode}): {detail}")
    return result


def locate_tools() -> tuple[str, str]:
    """Prefer explicit overrides, then isolated tools, then existing PATH tools."""
    found = []
    for name in ("ffmpeg", "ffprobe"):
        override = os.environ.get(f"FPVSESH_{name.upper()}")
        if override:
            candidate = Path(override).expanduser().resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"FPVSESH_{name.upper()} is not a file: {candidate}")
            found.append(str(candidate))
            continue
        executable = name + (".exe" if os.name == "nt" else "")
        # This tested build supports the installed NVENC driver. A newer
        # downloaded benchmark build can require a newer NVIDIA API.
        pinned = PROJECT_ROOT / "tools" / "ffmpeg-7.1.1" / "ffmpeg-7.1.1-full_build" / "bin" / executable
        local = ([pinned] if pinned.is_file() else
                 sorted((PROJECT_ROOT / "tools").glob(f"**/{executable}")))
        if local:
            found.append(str(local[0].resolve()))
        elif shutil.which(name):
            found.append(str(Path(shutil.which(name)).resolve()))
        else:
            raise FileNotFoundError(f"{name} is missing. Run the project's setup script.")
    return found[0], found[1]


def _number(value: Any, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _rational(value: Any, default="0/1") -> str:
    try:
        result = Fraction(str(value))
        return f"{result.numerator}/{result.denominator}"
    except (ValueError, ZeroDivisionError):
        return default


def _checked_source(path) -> Path:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Input must be a media file: {source}")
    if source.stat().st_size == 0:
        raise ValueError(f"Input is empty: {source}")
    return source


def sha256_file(path: str | Path) -> str:
    """Hash every byte with bounded memory; detect changes during hashing."""
    source = _checked_source(path)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {source}")
    return digest.hexdigest()


def probe(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    """Probe the first real video stream without editing the media.

    Missing color tags remain unknown. ``hdr`` is an explicit guard against
    silently treating PQ/HLG material as SDR. This function is not a decode test.
    """
    source = _checked_source(path)
    _, ffprobe = locate_tools()
    result = run([ffprobe, "-v", "error", "-show_streams", "-show_format",
                  "-of", "json", str(source)], timeout=120)
    raw = json.loads(result.stdout)
    streams = raw.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise ValueError(f"No usable video stream: {source}")
    video = videos[0]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    container = raw.get("format", {})
    duration = _number(video.get("duration"), _number(container.get("duration"), 0.0))
    if duration <= 0:
        raise ValueError(f"Video has no measurable positive duration: {source}")
    fps = _rational(video.get("avg_frame_rate"))
    nominal_fps = _rational(video.get("r_frame_rate"))
    if Fraction(fps) <= 0:
        fps = nominal_fps
    if Fraction(fps) <= 0:
        raise ValueError(f"Video has no usable frame rate: {source}")
    rotation = _number(video.get("tags", {}).get("rotate"), 0)
    for side_data in video.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = _number(side_data["rotation"], rotation)
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    if not width or not height:
        raise ValueError(f"Video has invalid dimensions: {source}")
    color = {key: video.get(key, "unknown") for key in
             ("color_range", "color_space", "color_transfer", "color_primaries",
              "chroma_location")}
    stat = source.stat()
    digest = sha256_file(source) if include_hash else None
    frame_count = _number(video.get("nb_frames"))
    frame_count = int(frame_count) if frame_count is not None else None
    result = {
        "source": str(source), "filename": source.name,
        "duration": duration, "width": width, "height": height,
        "display_width": height if abs(rotation) % 180 == 90 else width,
        "display_height": width if abs(rotation) % 180 == 90 else height,
        "sample_aspect_ratio": video.get("sample_aspect_ratio", "unknown"),
        "display_aspect_ratio": video.get("display_aspect_ratio", "unknown"),
        "fps": fps, "nominal_fps": nominal_fps,
        "fps_float": float(Fraction(fps)), "frame_count": frame_count,
        "frame_count_is_estimate": frame_count is None,
        "estimated_frame_count": round(duration * float(Fraction(fps))),
        "audio": bool(audio_streams), "audio_streams": audio_streams,
        "color": color, **color,
        "hdr": color["color_transfer"] in HDR_TRANSFERS,
        "pix_fmt": video.get("pix_fmt", "unknown"),
        "bit_depth": int(_number(video.get("bits_per_raw_sample"), 0)),
        "codec": video.get("codec_name", "unknown"),
        "profile": video.get("profile", "unknown"), "rotation": rotation,
        "time_base": _rational(video.get("time_base")),
        "start_time": _number(video.get("start_time"),
                              _number(container.get("start_time"), 0.0)),
        "start_pts": video.get("start_pts"), "duration_ts": video.get("duration_ts"),
        "video_stream_index": video["index"], "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "sha256": digest,
        "identity": {"sha256": digest, "size_bytes": stat.st_size},
        "vfr_suspected": fps != nominal_fps,
        "timestamp_scan_performed": False,
        "probe_warnings": result.stderr.strip(), "raw": raw,
    }
    return result


def inspect_timestamps(path: str | Path, *, timeout: float | None = 1800,
                       include_pts: bool = False) -> dict[str, Any]:
    """Decode all video frames and examine their integer presentation timestamps.

    Container tick rounding of one tick is allowed for constant-rate media.
    B-frame packet decode-order timestamps are deliberately not confused with
    frame presentation order. Missing/corrupt frames and PTS gaps are reported.
    """
    metadata = probe(path, include_hash=False)
    _, ffprobe = locate_tools()
    result = run([
        ffprobe, "-v", "error", "-select_streams", str(metadata["video_stream_index"]),
        "-show_frames", "-show_entries", "frame=pts,best_effort_timestamp,pkt_duration",
        "-of", "compact=p=0:nk=0", metadata["source"],
    ], timeout=timeout, check=False)
    pts = []
    missing = 0
    for line in result.stdout.splitlines():
        values = dict(part.split("=", 1) for part in line.split("|") if "=" in part)
        if not any(key in values for key in ("pts", "best_effort_timestamp", "pkt_duration")):
            continue
        value = values.get("pts", values.get("best_effort_timestamp"))
        try:
            pts.append(int(value))
        except (TypeError, ValueError):
            try:
                pts.append(int(values.get("best_effort_timestamp")))
            except (TypeError, ValueError):
                missing += 1
    deltas = [right - left for left, right in zip(pts, pts[1:])]
    positive = [delta for delta in deltas if delta > 0]
    median = statistics.median(positive) if positive else 0
    time_base = Fraction(metadata["time_base"])
    bad = sum(delta < 0 for delta in deltas)
    duplicates = sum(delta == 0 for delta in deltas)
    gap_threshold = max(median * 1.5, median + 1)
    gaps = [
        {"after_frame": index, "start_seconds": float(pts[index] * time_base),
         "delta_seconds": float(delta * time_base)}
        for index, delta in enumerate(deltas) if median and delta > gap_threshold
    ]
    varied = bool(positive) and (max(positive) - min(positive) > 1)
    error_lines = result.stderr.strip().splitlines()
    scan = {
        "frame_count": len(pts) + missing, "timestamped_frame_count": len(pts),
        "first_pts": pts[0] if pts else None, "last_pts": pts[-1] if pts else None,
        "time_base": metadata["time_base"],
        "first_time": float(pts[0] * time_base) if pts else None,
        "last_time": float(pts[-1] * time_base) if pts else None,
        "monotonic": bool(pts) and bad == 0,
        "strictly_monotonic": bool(pts) and bad == 0 and duplicates == 0 and missing == 0,
        "non_monotonic_count": bad, "duplicate_pts_count": duplicates,
        "missing_pts_count": missing, "vfr": varied or bool(duplicates),
        "median_step_seconds": float(median * time_base),
        "max_gap_seconds": float(max(positive, default=0) * time_base),
        "delta_tick_counts": {str(key): count for key, count in Counter(deltas).most_common(20)},
        "gap_count": len(gaps), "gaps": gaps[:100],
        "scan_complete": result.returncode == 0 and len(pts) > 0,
        "decode_error_count": len(error_lines), "decode_errors": error_lines[:30],
        "corruption_detected": result.returncode != 0 or bool(error_lines),
        "method": "Full decoded-frame scan of integer presentation timestamps; one-tick rounding tolerated.",
    }
    if include_pts:
        scan["pts"] = pts
    return scan


def choose_fps(probes: Sequence[dict]) -> str:
    """Preserve uniform 59.94/60; mixed sessions use true 60 without time warping."""
    if not probes:
        raise ValueError("At least one source is required to choose a timeline rate.")
    rates = [Fraction(item["fps"]) for item in probes]
    if all(rate == Fraction(60000, 1001) for rate in rates):
        return "60000/1001"
    return "60/1"


def fps_decision(probes: Sequence[dict]) -> dict[str, Any]:
    fps = choose_fps(probes)
    conversions = []
    for item in probes:
        if Fraction(item["fps"]) != Fraction(fps):
            conversions.append({
                "source": item.get("source", item.get("filename")),
                "source_fps": item["fps"], "output_fps": fps,
                "method": "Timestamp-based frame duplication/drop; preserve real-time duration; no interpolation.",
            })
    return {"fps": fps, "conversions": conversions,
            "reason": ("Preserve the exact uniform source rate." if not conversions else
                       "Use a consistent true-60 timeline for mixed/non-60 sources; preserve source timing.")}


def hardware_diagnostics() -> dict[str, Any]:
    """Read local hardware facts. No accounts, environment dump, serials or upload."""
    disk = shutil.disk_usage(PROJECT_ROOT)
    result = {
        "platform": platform.system(), "os_version": platform.version(),
        "architecture": platform.machine(), "logical_cpu_count": os.cpu_count(),
        "cpu": platform.processor(), "ram_bytes": None,
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "gpus": [], "gpu_access_proven": False,
        "note": "Enumeration is not proof of working GPU inference or encoding; see measured backend tests.",
    }
    if os.name == "nt":
        script = (
            "$taskCpu = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors); "
            "$taskRam = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
            "$taskGpu = @(Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,VideoProcessor); "
            "@{cpu=$taskCpu;ram_bytes=$taskRam;gpus=$taskGpu} | ConvertTo-Json -Depth 5 -Compress"
        )
        try:
            local = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], timeout=30)
            result.update(json.loads(local.stdout))
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            result["enumeration_warning"] = str(exc)
    elif hasattr(os, "sysconf"):
        try:
            result["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        queried = run([nvidia_smi, "--query-gpu=name,memory.total,driver_version",
                       "--format=csv,noheader,nounits"], timeout=30, check=False)
        if queried.returncode == 0:
            result["nvidia"] = []
            for line in queried.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 3:
                    result["nvidia"].append({"name": parts[0], "vram_mib": _number(parts[1]),
                                             "driver_version": parts[2]})
    try:
        ffmpeg, ffprobe = locate_tools()
        result["ffmpeg"] = {"path": ffmpeg, "version": run([ffmpeg, "-version"], timeout=20).stdout.splitlines()[0]}
        result["ffprobe"] = ffprobe
    except (OSError, RuntimeError):
        result["ffmpeg"] = None
    return result


def validate_output(path: str | Path, expected_frames: int, fps: str,
                    width: int, height: int, *, log_file=None) -> dict[str, Any]:
    """Verify structure, full decoding, frame PTS/count and possible black gaps.

    Black intervals are review flags, not automatic failures: legitimate dark
    footage exists. Original source audio cannot be judged semantically here;
    present audio streams are checked for gross start/end alignment only.
    """
    if expected_frames <= 0:
        raise ValueError("Expected frame count must be positive.")
    metadata = probe(path)
    timestamps = inspect_timestamps(path)
    ffmpeg, _ = locate_tools()
    decoded = run([
        ffmpeg, "-hide_banner", "-nostdin", "-v", "info", "-xerror",
        "-err_detect", "explode", "-i", metadata["source"], "-map", "0:v:0",
        "-map", "0:a?", "-vf", "blackdetect=d=0.10:pix_th=0.02:pic_th=0.98",
        "-f", "null", "-",
    ], log_file=log_file, timeout=max(600, metadata["duration"] * 20), check=False)
    black_intervals = []
    for match in re.finditer(r"black_start:([\d.eE+-]+)\s+black_end:([\d.eE+-]+)\s+black_duration:([\d.eE+-]+)", decoded.stderr):
        black_intervals.append({"start": float(match[1]), "end": float(match[2]),
                                "duration": float(match[3]), "review_required": True})
    expected_duration = float(Fraction(expected_frames, 1) / Fraction(fps))
    errors, warnings = [], []
    if (metadata["width"], metadata["height"]) != (width, height):
        errors.append(f"Dimensions are {metadata['width']}x{metadata['height']}; expected {width}x{height}.")
    if Fraction(metadata["fps"]) != Fraction(fps):
        errors.append(f"Frame rate is {metadata['fps']}; expected {fps}.")
    if timestamps["frame_count"] != expected_frames:
        errors.append(f"Decoded {timestamps['frame_count']} frames; expected {expected_frames}.")
    if not timestamps["strictly_monotonic"]:
        errors.append("Video presentation timestamps are missing, repeated, or non-monotonic.")
    if timestamps["vfr"] or timestamps["gap_count"]:
        errors.append("Output timestamps have varying frame intervals or missing-frame gaps.")
    if not timestamps["scan_complete"] or timestamps["decode_error_count"]:
        errors.append("Full frame inspection reported decoding errors or did not complete.")
    if decoded.returncode:
        errors.append("Full video/audio decode failed.")
    duration_tolerance = max(float(1 / Fraction(fps)), 0.005)
    if abs(metadata["duration"] - expected_duration) > duration_tolerance:
        errors.append(f"Video duration {metadata['duration']:.6f}s differs from timeline {expected_duration:.6f}s.")
    if abs(timestamps["first_time"] or 0) > duration_tolerance:
        errors.append("Output does not start at timeline time zero.")
    if black_intervals:
        warnings.append("Possible black gaps detected; inspect listed intervals because naturally dark shots can trigger this check.")
    audio_alignment = []
    for stream in metadata["audio_streams"]:
        start = _number(stream.get("start_time"), 0.0)
        duration = _number(stream.get("duration"))
        aligned = abs(start) <= 0.12 and duration is not None and abs(start + duration - expected_duration) <= 0.12
        audio_alignment.append({"stream_index": stream["index"], "start": start,
                                "duration": duration, "aligned_within_120ms": aligned})
        if not aligned:
            errors.append("Audio stream start/end differs from the video timeline by more than 120 ms or is unavailable.")
    return {
        "passed": not errors, "errors": errors, "warnings": warnings,
        "expected_frames": expected_frames, "expected_duration": expected_duration,
        "probe": metadata, "timestamps": timestamps,
        "full_decode": {"passed": decoded.returncode == 0, "returncode": decoded.returncode,
                        "error_tail": decoded.stderr[-4000:] if decoded.returncode else ""},
        "black_intervals": black_intervals, "audio_alignment": audio_alignment,
        "visual_quality_reviewed": False,
    }
