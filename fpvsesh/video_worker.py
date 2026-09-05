"""Isolated, offline Qwen video inference with bounded frames and GPU memory."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

# No network is used during inference, including implicit model-hub lookups.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
from .control import Cancelled, check_control
from .video_understanding import _save, combine_evidence, interpret, prompt
from .rotation_witness import inspect_rotation


def decode_window(source, window, profile, ffmpeg, checkpoint=lambda: None):
    """FFmpeg handles source timing/rotation; aspect is fitted into a square."""
    duration = window["end"]-window["start"]
    expected = math.ceil(duration*profile["sample_fps"]-1e-6)
    if not 2 <= expected <= profile["max_frames"]:
        raise ValueError("Video window must fit the complete sampled-frame budget")
    # Trim source timestamps before resampling. Output -t can quantize a
    # fractional window down and discard its last valid sampled frame.
    filters = (f"trim=duration={duration},fps={profile['sample_fps']}:start_time=0:eof_action=pass,"
               "scale=256:256:force_original_aspect_ratio=decrease:force_divisible_by=2,"
               "pad=256:256:(ow-iw)/2:(oh-ih)/2,setsar=1")
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", str(window["start"]),
            "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", filters, "-frames:v", str(expected), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    started = time.monotonic()
    try:
        while True:
            checkpoint()
            try:
                raw, error = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic()-started > 120:
                    raise RuntimeError("Timed out decoding a video observation window")
        if process.returncode:
            raise RuntimeError(error.decode("utf-8", errors="replace")[-1000:])
        count, remainder = divmod(len(raw), 256*256*3)
        if remainder or count != expected:
            raise ValueError(f"Video window ended early or had incomplete frames: expected {expected}, decoded {count}")
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(count, 256, 256, 3).copy()
        return frames, [index/profile["sample_fps"] for index in range(count)]
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


def run(config):
    from .runtime_dlls import prepare_torch_dlls
    prepare_torch_dlls()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
    control = Path(config["control"])
    checkpoint = lambda: check_control(control)
    checkpoint()
    if not torch.cuda.is_available():
        raise RuntimeError("The installed video model requires the local CUDA GPU; CPU inference is not enabled")
    if torch.cuda.get_device_properties(0).total_memory < 7*1024**3:
        raise RuntimeError("The tested video profile requires at least 8 GB-class GPU memory")
    torch.set_num_threads(4)
    model_path = Path(config["model_folder"])
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        trust_remote_code=False).to("cuda").eval()
    checkpoint()

    class ControlStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            checkpoint()
            token_times.append(time.monotonic())
            return False

    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt()}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    profile = config["profile"]
    processor.video_processor.size = {"shortest_edge": 128*128, "longest_edge": profile["pixel_budget"]}
    for index, window in enumerate(config["windows"]):
        checkpoint()
        started = time.monotonic()
        frames, times = decode_window(window["source"], window["window"], profile, config["ffmpeg"], checkpoint)
        witness = inspect_rotation(window["source"], **window["window"], ffmpeg=config["ffmpeg"], checkpoint=checkpoint)
        decoded = time.monotonic()
        metadata = {"total_num_frames": len(frames), "fps": profile["sample_fps"],
                    "frames_indices": list(range(len(frames))), "duration": len(frames)/profile["sample_fps"]}
        inputs = processor(text=[text], videos=[frames], video_metadata=[metadata],
                           do_sample_frames=False, return_tensors="pt").to("cuda")
        del frames
        prepared = time.monotonic()
        token_times = []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=192, do_sample=False,
                                    stopping_criteria=StoppingCriteriaList([ControlStop()]),
                                    temperature=None, top_p=None, top_k=None)
        generated = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        observation = combine_evidence(interpret(generated, **window["window"]), witness)
        observation["motion_support"] = window.get("motion", {})
        record = {"signature": window["signature"], "event": observation, "inference_demonstrated": True,
                  "rotation_witness": witness,
                  "sampled_frames": len(times), "sample_times": times, "raw_response": generated[:6000],
                  "elapsed_seconds": round(time.monotonic()-started, 3),
                  "decode_seconds": round(decoded-started, 3), "prepare_seconds": round(prepared-decoded, 3),
                  "generate_seconds": round(time.monotonic()-prepared, 3),
                  "first_token_seconds": round(token_times[0]-prepared, 3) if token_times else None,
                  "generated_tokens": len(token_times),
                  "peak_reserved_gpu_mib": round(torch.cuda.max_memory_reserved()/1024**2),
                  "peak_allocated_gpu_mib": round(torch.cuda.max_memory_allocated()/1024**2),
                  "device": torch.cuda.get_device_name(0), "dtype": "bfloat16", "attention": "sdpa"}
        _save(window["output"], record)
        _save(config["progress"], {"completed": index+1, "total": len(config["windows"]), "label": observation["label"]})
        del inputs, output
        torch.cuda.empty_cache()
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        return run(json.loads(Path(args.config).read_text(encoding="utf-8")))
    except Cancelled:
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
