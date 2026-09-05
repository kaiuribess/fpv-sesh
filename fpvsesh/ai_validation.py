"""Reproducible local GPU sample validation; no visual-quality certification."""
from datetime import datetime, timezone
from fractions import Fraction
import math
import os
from pathlib import Path
import uuid

from .analysis import save_json
from . import cuda_backend
from .enhance import _detected_gpu, _fit_dimensions, _probe
from .media import locate_tools, probe, validate_output

ROOT = Path(__file__).resolve().parents[1]


def validate_ai(source, start=0, seconds=2, log=print):
    if not math.isfinite(start) or start < 0 or not math.isfinite(seconds) or not .5 <= seconds <= 5:
        raise ValueError("Choose a nonnegative start and a sample between 0.5 and 5 seconds")
    for directory in ("logs", "cache"):
        (ROOT / directory).mkdir(exist_ok=True)
    source = Path(source).expanduser().resolve(strict=True)
    meta = probe(source)
    if meta["hdr"] or any(meta.get(tag) not in ("bt709", "unknown") for tag in ("color_space", "color_primaries", "color_transfer")):
        raise ValueError("AI sample validation currently requires SDR Rec.709 footage")
    if meta.get("color_range") not in ("tv", "unknown") or meta.get("sample_aspect_ratio") not in ("1:1", "1/1", "unknown", "N/A"):
        raise ValueError("AI sample validation requires limited-range, square-pixel footage")
    if start + seconds > meta["duration"]:
        raise ValueError("Sample extends beyond the source recording")
    expected_signature = cuda_backend.signature()
    gpu = _detected_gpu()
    if not gpu.get("uuid"):
        raise RuntimeError("A compatible NVIDIA GPU could not be identified")
    ffmpeg, ffprobe = locate_tools()
    source_probe = _probe(source, Path(ffprobe))
    fps = Fraction(meta["fps"])
    frames = round(seconds * fps)
    options = {"ffmpeg": ffmpeg, "start": start, "duration": float(frames / fps),
               "frames": frames, "fps": str(fps), "width": 3840, "height": 2160,
               "codec": "hevc", "grade": "", "rate_conversion": True}
    folder = ROOT / "logs" / ("ai-validation-" + uuid.uuid4().hex[:12])
    folder.mkdir()
    output = folder / "sample.mp4"
    lock = (ROOT / "cache/run.lock").open("a+b")
    try:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another FPV Sesh render is running") from exc
        (ROOT / "cache/control.json").unlink(missing_ok=True)
        log(f"Validating {frames} real frames using the installed CUDA model")
        record = cuda_backend.render(source, output, options, source_probe, _fit_dimensions(source_probe, 3840, 2160), log)
        verification = validate_output(output, frames, str(fps), 3840, 2160)
        save_json(folder / "verification.json", verification)
        if not verification["passed"]:
            raise RuntimeError("AI sample failed decoded-frame verification: " + str(verification["errors"]))
        if cuda_backend.signature() != expected_signature or _detected_gpu() != gpu:
            raise RuntimeError("GPU, code or model changed during validation; retry the sample")
        result = {"passed": True, "gpu": gpu, "signature": expected_signature,
                  "validated_utc": datetime.now(timezone.utc).isoformat(), "sample": str(output),
                  "source_sha256": meta["sha256"], "frames": frames,
                  "fps": frames / max(record["elapsed_seconds"], .001),
                  "peak_total_gpu_memory_mib": record.get("peak_total_gpu_memory_mib"),
                  "scope": "Real local inference and output integrity; inspect the sample yourself for texture and temporal quality"}
        save_json(cuda_backend.VALIDATION, result)
        log("AI sample passed. Inspect the sample for natural detail before choosing AI for a full edit.")
        return result
    finally:
        lock.close()
