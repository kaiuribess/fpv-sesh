"""Main-app adapter for the isolated and locally validated CUDA worker."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

from .control import Cancelled
from .media import locate_tools

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-ai/Scripts/python.exe"
VALIDATION = ROOT / "logs/cuda-enhancement-tested.json"
PROFILE = {"ai_model": "RealESRGAN_x2plus", "ai_blend": .4, "ai_denoise": .2,
           "ai_tile": 768, "cq": 16, "encoder_preset": "p7"}


def signature(*, ffmpeg=None, ffprobe=None):
    """Fingerprint the binaries the caller will use, including explicit overrides."""
    paths = [ROOT / "fpvsesh/ai_models.py", ROOT / "fpvsesh/ai_worker.py",
             ROOT / "models/real-esrgan-cuda/RealESRGAN_x2plus.pth",
             ROOT / ".venv-ai/Lib/site-packages/torch/version.py"]
    hashes = {}
    for path in paths:
        with path.open("rb") as stream:
            hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.file_digest(stream, "sha256").hexdigest()
    if ffmpeg is None or ffprobe is None:
        default_ffmpeg, default_ffprobe = locate_tools()
        ffmpeg = default_ffmpeg if ffmpeg is None else ffmpeg
        ffprobe = default_ffprobe if ffprobe is None else ffprobe
    tools = {}
    for name, executable in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        with Path(executable).open("rb") as stream:
            tools[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"profile": PROFILE, "files": hashes, "tools": tools}


def status(gpu, *, ffmpeg=None, ffprobe=None):
    try:
        record = json.loads(VALIDATION.read_text(encoding="utf-8"))
        available = (PYTHON.is_file() and record.get("passed") is True and
                     record.get("gpu") == gpu and record.get("signature") == signature(ffmpeg=ffmpeg, ffprobe=ffprobe))
    except (OSError, ValueError):
        record, available = {}, False
    return {"available": available, "profile": PROFILE,
            "fps": record.get("fps") if available else None,
            "peak_total_gpu_memory_mib": record.get("peak_total_gpu_memory_mib") if available else None}


def render(source, destination, options, source_probe, fit_dimensions, log):
    """Stream progress from one child and propagate safe worker cancellation."""
    if not PYTHON.is_file():
        raise RuntimeError("The isolated CUDA enhancement environment is missing")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    token = uuid.uuid4().hex[:12]
    config_path = ROOT / "cache" / f"cuda-{token}.json"
    log_path = ROOT / "logs" / f"cuda-{token}.log"
    width, height = source_probe["width"], source_probe["height"]
    rotation = next((r["rotation"] for r in source_probe.get("side_data_list", []) if "rotation" in r),
                    source_probe.get("tags", {}).get("rotate", 0))
    if abs(float(rotation)) % 180 == 90:
        width, height = height, width
    config = {**options, **PROFILE, "source": str(source), "output": str(destination),
              "source_width": width, "source_height": height,
              "content_width": fit_dimensions[0], "content_height": fit_dimensions[1]}
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    lines = queue.Queue()
    process = None
    worker = None
    started = time.monotonic()
    gpu_samples = []
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "fpvsesh.ai_worker", "--config", str(config_path)],
                                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace",
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            def read_output():
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    lines.put(line)
            worker = threading.Thread(target=read_output, daemon=True)
            worker.start()
            while process.poll() is None or not lines.empty():
                try:
                    line = lines.get(timeout=.3)
                except queue.Empty:
                    continue
                try:
                    progress = json.loads(line)
                except ValueError:
                    continue
                if progress.get("stage") in ("paused", "resumed"):
                    log(progress.get("message", progress["stage"]))
                elif progress.get("state"):
                    log(str(progress["state"]))
                elif "frames" in progress:
                    log(f"Restoring natural detail: {progress['frames']}/{progress['total']} frames ({progress.get('fps', 0):.2f} fps)")
                    try:
                        measured = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                                  capture_output=True, text=True, timeout=3,
                                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        gpu_samples.append(int(measured.stdout.strip().splitlines()[0]))
                    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                        pass
            worker.join(timeout=5)
            if worker.is_alive():
                raise RuntimeError("CUDA worker output did not close")
            if process.returncode == 75:
                raise Cancelled("Cancelled after a completed AI frame; finished segments remain cached")
            if process.returncode:
                raise RuntimeError("CUDA detail restoration failed; " + log_path.read_text(encoding="utf-8")[-2000:])
        result_path = destination.with_suffix(destination.suffix + ".ai.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_path.unlink()
        result.update({"source_path": str(source), "output_path": str(destination),
                       "start": options["start"], "duration": options["duration"], "warnings": [],
                       "elapsed_seconds": round(time.monotonic()-started, 3),
                       "scaling_note": "Native 2x realistic CUDA restoration, fixed 40% model / 60% source-scale blend; no invented motion.",
                       "frame_rate_conversion": options.get("rate_conversion", False),
                       "peak_total_gpu_memory_mib": max(gpu_samples, default=None),
                       "memory_measurement": "Total GPU usage sampled at worker progress reports; includes other applications and may miss brief peaks.",
                       "log_path": str(log_path),
                       "model_signature": signature(ffmpeg=options.get("ffmpeg"), ffprobe=options.get("ffprobe"))})
        return result
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            if process.stdout is not None:
                process.stdout.close()
        config_path.unlink(missing_ok=True)
