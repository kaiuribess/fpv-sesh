"""Isolated CUDA restoration worker: original frames -> model -> one encode.

Run with .venv-ai, independently of the desktop app's small environment.
Only selected contiguous source frames enter the model. Buffers hold one frame.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from fractions import Fraction

import cv2
import numpy as np

from .ai_models import Restorer
from .control import Cancelled, check_control

ROOT = Path(__file__).resolve().parents[1]
CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _read_frame(stream, count):
    parts = bytearray()
    while len(parts) < count:
        part = stream.read(count - len(parts))
        if not part:
            break
        parts.extend(part)
    if len(parts) != count:
        raise RuntimeError(f"Source decode ended with {len(parts)} of {count} frame bytes")
    return parts


def _encoder_plan(config, *, software=False):
    """Describe the actual encoding path; CPU CRF and NVENC CQ are distinct."""
    codec = config.get("codec", "hevc")
    quality = float(config.get("cq", 16))
    preset = config.get("encoder_preset", "p7")
    if codec not in ("hevc", "h264") or not math.isfinite(quality) or not 0 <= quality <= 51:
        raise ValueError("AI encoding requires HEVC or H.264 and a quality value from 0 to 51")
    if preset not in {f"p{number}" for number in range(1, 8)}:
        raise ValueError("Invalid NVENC preset")
    name = ("libx265" if codec == "hevc" else "libx264") if software else ("hevc_nvenc" if codec == "hevc" else "h264_nvenc")
    pixel = ("yuv420p10le" if software else "p010le") if codec == "hevc" else "yuv420p"
    args = ["-c:v", name, "-preset", "medium" if software else preset]
    if software:
        args += ["-crf", str(quality)]
        color = "colorprim=bt709:transfer=bt709:colormatrix=bt709"
        args += (["-x265-params", color + ":range=limited:log-level=error"] if codec == "hevc"
                 else ["-x264-params", color])
    else:
        args += ["-tune", "hq", "-rc", "vbr", "-cq", str(quality), "-b:v", "0",
                 "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32", "-multipass", "fullres"]
    if codec == "hevc":
        args += ["-profile:v", "main10", "-tag:v", "hvc1"]
        if not software:
            args += ["-tier", "high", "-maxrate", "120M"]
    return {"encoder": name, "encoder_preset": "medium" if software else preset,
            "rate_control": "crf" if software else "cq", "quality_value": quality,
            "cq": None if software else quality, "crf": quality if software else None,
            "encoder_fallback": software, "pixel_format": pixel, "args": args}


def _select_encoder(ffmpeg, config, checkpoint, log_file):
    """Actually encode three synthetic frames before loading the CUDA model.

    This probes codec initialization, profile and pixel-format support on the
    installed driver. It does not substitute for validating the finished video.
    No decoded source frame or model result is involved in the fallback.
    """
    fps = str(Fraction(config["fps"]))
    print(json.dumps({"state": "Checking video encoder before AI restoration"}), flush=True)
    with Path(log_file).open("w", encoding="utf-8") as handle:
        for software in (False, True):
            checkpoint()
            plan = _encoder_plan(config, software=software)
            command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                       "-pix_fmt", "bgr24", "-video_size", "320x180", "-framerate", fps,
                       "-i", "pipe:0", "-an", "-vf",
                       "scale=in_range=pc:out_range=tv:in_color_matrix=bt709:out_color_matrix=bt709,format=" + plan["pixel_format"],
                       "-frames:v", "3", *plan["args"], "-color_primaries", "bt709",
                       "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
                       "-r", fps, "-fps_mode", "cfr", "-f", "null", "-"]
            handle.write(json.dumps(command) + "\n")
            handle.flush()
            try:
                checked = subprocess.run(command, input=bytes(320 * 180 * 3 * 3),
                                         capture_output=True, timeout=20, creationflags=CREATE_FLAGS)
                detail = checked.stderr.decode("utf-8", errors="replace")
                passed = checked.returncode == 0
            except subprocess.TimeoutExpired:
                detail, passed = "Encoder capability check timed out after 20 seconds", False
            except OSError as error:
                raise RuntimeError("Could not start the video encoder capability check") from error
            handle.write(detail + "\n")
            handle.flush()
            checkpoint()
            if passed:
                plan["warnings"] = (["GPU video encoding is unavailable with the installed encoder/driver. "
                                     f"Using {plan['encoder']} CPU encoding; CUDA restoration remains enabled."]
                                    if software else [])
                print(json.dumps({"state": plan["warnings"][0] if software else
                                  f"Video encoder ready: {plan['encoder']}"}), flush=True)
                return plan
    raise RuntimeError("GPU and CPU video encoder checks both failed; see the encoder-probe log")


def render(config):
    source = Path(config["source"]).resolve(strict=True)
    output = Path(config["output"]).resolve()
    if source == output or not output.is_relative_to(ROOT):
        raise ValueError("CUDA output must be an application-owned file distinct from its source")
    ffmpeg = Path(config["ffmpeg"])
    frames = int(config["frames"])
    fps = Fraction(config["fps"])
    w, h = int(config["source_width"]), int(config["source_height"])
    fit_w, fit_h = int(config["content_width"]), int(config["content_height"])
    width, height = int(config["width"]), int(config["height"])
    blend = float(config.get("ai_blend", .65))
    if not (0 <= blend <= 1) or frames <= 0 or fps <= 0:
        raise ValueError("Invalid restoration blend, frame count or rate")
    if not math.isfinite(float(config["start"])) or float(config["start"]) < 0:
        raise ValueError("AI source start must be finite and nonnegative")
    if (min(w, h, fit_w, fit_h, width, height) <= 0 or any(v % 2 for v in (fit_w, fit_h, width, height))
            or fit_w > width or fit_h > height or abs(fit_w/fit_h - w/h) > 2/min(fit_w, fit_h)):
        raise ValueError("AI content dimensions must fit the canvas and preserve source aspect")
    completed = 0
    def checkpoint():
        def report(stage):
            print(json.dumps({"stage": stage, "frames": completed, "total": frames,
                              "message": "Paused between frames" if stage == "paused" else "Resuming enhancement"}), flush=True)
        check_control(ROOT / "cache/control.json", on_pause=lambda: report("paused"),
                      on_resume=lambda: report("resumed"))
    checkpoint()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoding = _select_encoder(ffmpeg, config, checkpoint, output.with_suffix(".encoder-probe.log"))
    restorer = Restorer(config["ai_model"], ROOT / "models/real-esrgan-cuda",
                        tile=int(config.get("ai_tile", 384)),
                        denoise=float(config.get("ai_denoise", .2)), cudnn_benchmark=True)
    decode = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-ss", str(config["start"]),
              "-i", str(source), "-map", "0:v:0", "-an", "-sn"]
    if config.get("rate_conversion"):
        decode += ["-vf", f"fps={fps}:round=near"]
    decode += ["-frames:v", str(frames), "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    pixel = encoding["pixel_format"]
    vf = ["scale=in_range=pc:out_range=tv:in_color_matrix=bt709:out_color_matrix=bt709",
          "format=yuv444p10le"]
    if config.get("grade"):
        vf.append(config["grade"])
    vf += [f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black", "setsar=1", f"format={pixel}"]
    encode = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
              "-pix_fmt", "bgr24", "-video_size", f"{fit_w}x{fit_h}", "-framerate", str(fps),
              "-i", "pipe:0", "-an", "-vf", ",".join(vf), "-frames:v", str(frames), *encoding["args"]]
    encode += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
               "-color_range", "tv", "-r", str(fps), "-fps_mode", "cfr", "-movflags", "+faststart",
               "-video_track_timescale", str(fps.numerator), str(output)]
    started = time.monotonic()
    decoder = encoder = None
    with output.with_suffix(".decode.log").open("w") as decode_log, output.with_suffix(".encode.log").open("w") as encode_log:
        try:
            decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=decode_log, creationflags=CREATE_FLAGS)
            encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=encode_log, creationflags=CREATE_FLAGS)
            last_report = started
            for index in range(frames):
                checkpoint()
                raw = _read_frame(decoder.stdout, w * h * 3)
                original = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                restored = restorer.enhance(original, outscale=min(2, fit_h / h))
                if restored.shape[:2] != (fit_h, fit_w):
                    restored = cv2.resize(restored, (fit_w, fit_h), interpolation=cv2.INTER_LANCZOS4)
                if blend < 1:
                    conventional = cv2.resize(original, (fit_w, fit_h), interpolation=cv2.INTER_LANCZOS4)
                    restored = cv2.addWeighted(restored, blend, conventional, 1-blend, 0)
                encoder.stdin.write(restored.tobytes())
                completed = index + 1
                now = time.monotonic()
                if now - last_report >= 10 or index + 1 == frames:
                    print(json.dumps({"frames": index+1, "total": frames,
                                      "elapsed": round(now-started, 2),
                                      "fps": round((index+1)/(now-started), 3)}), flush=True)
                    last_report = now
            checkpoint()
            encoder.stdin.close()
            if encoder.wait(timeout=180) != 0:
                raise RuntimeError("CUDA enhanced video encoder failed; see its encode log")
            decoder.stdout.close()
            if decoder.wait(timeout=30) != 0:
                raise RuntimeError("Original selected-interval decode failed; see its decode log")
        finally:
            for process in (decoder, encoder):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if process is not None:
                    for stream in (process.stdin, process.stdout):
                        if stream is not None and not stream.closed:
                            try:
                                stream.close()
                            except OSError:
                                # A stopped encoder can reject a buffered pipe
                                # flush. Preserve the original cancel/error.
                                pass
            restorer.close()
    result = {"ai_inference": True, "backend": "real-esrgan-pytorch-cuda",
              **{key: value for key, value in encoding.items() if key != "args"},
              "model": config["ai_model"], "blend": blend,
              "denoise": config.get("ai_denoise", .2), "tile": config.get("ai_tile", 384),
              "fps": str(fps), "frames": frames, "elapsed_seconds": round(time.monotonic()-started, 3),
              "width": width, "height": height, "content_width": fit_w, "content_height": fit_h,
              "native_model_scale": 2 if config["ai_model"] == "RealESRGAN_x2plus" else 4}
    output.with_suffix(output.suffix + ".ai.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        render(json.loads(Path(args.config).read_text(encoding="utf-8")))
    except Cancelled as error:
        print(json.dumps({"stage": "cancelled", "message": str(error)}), flush=True)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
