"""Isolated optional scene inference process; source/proxy files stay read-only."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import cv2
from .control import Cancelled, check_control
from .vision_models import SceneModel


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    # Windows readers can briefly deny replacement while opening progress JSON.
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.02)


def run(config):
    model = SceneModel(config.get("model_folder"))
    total = sum(math.ceil(source["duration"]) for source in config["sources"])
    completed = 0
    for source in config["sources"]:
        cap = cv2.VideoCapture(source["proxy"])
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError("Cannot open the scene-analysis proxy")
            frames, times, samples, index, next_sample = [], [], [], 0, 0
            while True:
                check_control(Path(config["control"]))
                ok, frame = cap.read()
                if not ok:
                    break
                t = index / fps
                if t + 1e-5 >= next_sample and t < source["duration"]:
                    frames.append(frame)
                    times.append(round(t, 4))
                    next_sample += 1
                index += 1
                if len(frames) == 8:
                    samples.extend({"t": t, **value} for t, value in zip(times, model.predict(frames)))
                    completed += len(frames)
                    frames, times = [], []
                    save(config["progress"], {"completed": completed, "total": total, "source": source["source"]})
            if frames:
                samples.extend({"t": t, **value} for t, value in zip(times, model.predict(frames)))
                completed += len(frames)
            if index / fps < source["duration"] - max(.2, 2 / fps) or not samples:
                raise ValueError("Scene proxy decode ended before the source duration")
            save(source["output"], {"signature": source["signature"], "samples": samples,
                                     "coverage_seconds": index / fps, "proxy_fps": fps,
                                     "device": str(model.device), "model_hash": model.model_hash})
        finally:
            cap.release()
    save(config["progress"], {"completed": completed, "total": total})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        run(json.loads(Path(args.config).read_text(encoding="utf-8")))
    except Cancelled:
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
