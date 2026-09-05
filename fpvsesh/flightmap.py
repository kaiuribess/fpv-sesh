"""Reviewable motion maps with optional internet-pretrained scene context.

Scene recognition does not recognize named FPV tricks. Local confirmed labels
and measured motion estimates retain their own provenance and uncertainty.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import numpy as np
from .control import Cancelled

ROOT = Path(__file__).resolve().parents[1]
VERSION = 1
FEATURES = ["motion_median", "rotation_p90", "dx_median", "dy_median", "proximity_mean",
            "parallax_mean", "idle_fraction", "rotation_fraction"]


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".partial")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scene_model_status():
    folder = ROOT / "models/places365"
    manifest = _read(folder / "manifest.json", {})
    python = ROOT / ".venv-ai/Scripts/python.exe"
    try:
        assets = manifest["assets"]
        valid = bool(assets) and python.is_file() and all(
            (folder / entry["file"]).is_file() and
            (folder / entry["file"]).stat().st_size == entry["size_bytes"] and
            _hash(folder / entry["file"]) == entry["sha256"] for entry in assets)
    except (KeyError, OSError, TypeError):
        valid = False
    return {"available": valid, "name": "Places365 ResNet18", "method": "online-pretrained scene classifier",
            "model_hash": manifest.get("assets", [{}])[0].get("sha256") if manifest.get("assets") else None,
            "message": ("Internet-pretrained scene context is available; named tricks remain unverified."
                        if valid else "Motion estimates available. Optional online-pretrained scene model is not installed or failed integrity checks; run setup-vision.ps1."),
            "license": "CC BY (version unspecified by upstream)",
            "training": "Places365-Standard, approximately 1.8 million scene images / 365 classes"}


def _valid_samples(record, signature, duration):
    if not isinstance(record, dict) or record.get("signature") != signature:
        return False
    samples = record.get("samples")
    previous = -1
    try:
        coverage, fps = float(record["coverage_seconds"]), float(record["proxy_fps"])
        if not math.isfinite(coverage) or not math.isfinite(fps) or fps <= 0 or coverage <= 0:
            return False
        if abs(coverage-duration) > max(.2, 2/fps):
            return False
        if not isinstance(samples, list) or len(samples) != math.ceil(min(duration, coverage)-1e-6):
            return False
        for sample in samples:
            t = float(sample["t"])
            if not math.isfinite(t) or not previous < t < duration or abs(t - round(t)) > .1:
                return False
            previous = t
            if not isinstance(sample["scene"], str) or not 0 <= float(sample["score"]) <= 1:
                return False
            if not isinstance(sample.get("groups"), dict) or not sample.get("top_classes"):
                return False
            if not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in sample["groups"].values()):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _scene_samples(analyses, cache, event, checkpoint):
    for analysis in analyses:
        analysis.pop("scene_samples", None)
    status = scene_model_status()
    if not status["available"]:
        return status
    scene_cache = Path(cache) / "learning/scenes"
    scene_cache.mkdir(parents=True, exist_ok=True)
    pending, records = [], []
    code_hash = hashlib.sha256((_hash(ROOT / "fpvsesh/vision_models.py") + _hash(ROOT / "fpvsesh/vision_worker.py")).encode()).hexdigest()
    for analysis in analyses:
        checkpoint()
        proxy = Path(analysis.get("proxy", ""))
        if not proxy.is_file():
            continue
        signature = {"model": status["model_hash"], "code": code_hash,
                     "source": analysis["identity"], "proxy": _hash(proxy), "version": VERSION,
                     "duration": analysis["duration"]}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:32]
        output = scene_cache / f"{key}.json"
        record = _read(output, {})
        records.append((analysis, signature, output))
        if not _valid_samples(record, signature, analysis["duration"]):
            pending.append({"source": analysis["source"], "proxy": str(proxy.resolve()),
                            "duration": analysis["duration"], "signature": signature, "output": str(output.resolve())})
    if pending:
        token = uuid.uuid4().hex[:12]
        config_path = scene_cache / f"worker-{token}.json"
        progress_path = scene_cache / f"progress-{token}.json"
        config = {"sources": pending, "control": str((Path(cache) / "control.json").resolve()),
                  "progress": str(progress_path.resolve())}
        _save(config_path, config)
        process = None
        try:
            event("flight-map", 0, "Recognizing surroundings with the online-pretrained scene model")
            with tempfile.TemporaryFile(mode="w+b") as log:
                process = subprocess.Popen([str(ROOT / ".venv-ai/Scripts/python.exe"), "-m", "fpvsesh.vision_worker", "--config", str(config_path.resolve())],
                                           cwd=ROOT, stdout=log, stderr=log,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                last_progress = -1
                while process.poll() is None:
                    checkpoint()
                    progress = _read(progress_path, {})
                    if progress and progress.get("completed", 0) > last_progress:
                        last_progress = progress.get("completed", 0)
                        event("flight-map", last_progress / max(1, progress.get("total", 1)),
                              f"Scene context: {last_progress}/{progress.get('total', '?')} sampled seconds")
                    time.sleep(.1)
                if process.returncode == 75:
                    raise Cancelled("Scene mapping cancelled; completed source maps remain cached")
                if process.returncode:
                    log.seek(0)
                    raise RuntimeError(log.read().decode("utf-8", errors="replace")[-1500:])
        except Cancelled:
            raise
        except (OSError, RuntimeError) as error:
            status.update({"available": False, "message": "Scene inference unavailable; using motion estimates. " + str(error)[-500:]})
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            config_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
    count = 0
    for analysis, signature, output in records:
        record = _read(output, {})
        if _valid_samples(record, signature, analysis["duration"]):
            analysis["scene_samples"] = record["samples"]
            count += len(record["samples"])
    status["sampled_frames"] = count
    if count == 0 and status["available"]:
        status["message"] = "Scene model installed, but no valid proxy samples were available; map uses motion estimates."
    status["sampling"] = "one proxy frame per second; scene scores are uncalibrated softmax mass"
    return status


def _features(rows, start, end):
    chosen = [row for row in rows if start <= row["t"] < end]
    if not chosen:
        return None
    def values(key):
        return np.array([float(row.get(key, 0)) for row in chosen])
    motion, rotation = values("motion"), np.abs(values("rotation"))
    vector = [np.median(motion)/20, np.percentile(rotation, 90)/25,
              np.median(np.abs(values("dx")))/15, np.median(np.abs(values("dy")))/15,
              np.mean(values("proximity")), np.mean(values("parallax_confidence")),
              np.mean(motion < .4), np.mean(rotation >= 8)]
    return [round(float(np.clip(value, 0, 3)), 5) for value in vector] if np.isfinite(vector).all() else None


def _load_examples(path):
    existing = _read(path, {})
    if not isinstance(existing, dict) or existing.get("version") != VERSION or not isinstance(existing.get("examples"), list):
        return []
    valid = []
    for item in existing["examples"]:
        try:
            if (not isinstance(item, dict) or not isinstance(item["features"], list) or
                len(item["features"]) != len(FEATURES) or not np.isfinite(item["features"]).all() or
                not isinstance(item["source_identity"], str) or not item["source_identity"] or
                not isinstance(item["label"], str) or not item["label"].strip() or
                not math.isfinite(item["start"]) or not math.isfinite(item["end"]) or
                not 0 <= item["start"] < item["end"] or item.get("confidence") != 1 or
                item.get("method") != "user-confirmed" or not all(0 <= value <= 3 for value in item["features"])):
                continue
            if item.get("confirmation_bound") is not True:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        valid.append(item)
    return valid


def _dataset(analyses, labels, learning_dir):
    path = Path(learning_dir) / "confirmed-examples.json"
    examples = _load_examples(path)
    saved = _read(path, {})
    saved_rows = saved.get("examples", []) if isinstance(saved, dict) else []
    ignored = (len(saved_rows)-len(examples)) if isinstance(saved_rows, list) else 1
    warnings = ([f"Ignored {ignored} malformed local learning examples; valid confirmations remain available."]
                if ignored > 0 else [])
    source_map = {a["source"]: a for a in analyses}
    if isinstance(labels, dict):
        labels = labels.get("labels", [])
    if not isinstance(labels, list):
        raise ValueError("Flight labels must be a list of confirmed intervals")
    current = []
    for label in labels:
        if not isinstance(label, dict) or label.get("source") not in source_map:
            raise ValueError("Confirmed flight label references an unknown source")
        source = source_map[label["source"]]
        bound = label.get("source_identity")
        if (not isinstance(bound, str) or len(bound) != 64 or
                any(character not in "0123456789abcdef" for character in bound) or bound != source["identity"]):
            warnings.append(f"Skipped unbound or stale label for {Path(source['source']).name}; re-confirm it against the current source before learning.")
            continue
        start, end = float(label.get("start", -1)), float(label.get("end", -1))
        name = str(label.get("label", "")).strip()
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= source["duration"]):
            raise ValueError("Confirmed flight label is outside its source duration")
        if not name or len(name) > 100 or label.get("confidence", 1) != 1:
            raise ValueError("Flight learning accepts explicit confirmed labels with confidence 1 only")
        features = _features(source["rows"], start, end)
        if features is None:
            raise ValueError("Confirmed flight interval has no usable motion measurements")
        record = {"source_identity": source["identity"], "start": start, "end": end,
                  "label": name, "features": features, "method": "user-confirmed", "confidence": 1,
                  "confirmation_bound": True}
        current.append({**record, "source": source["source"]})
        examples = [item for item in examples if (item["source_identity"], item["start"], item["end"]) != (record["source_identity"], start, end)]
        examples.append(record)
    _save(path, {"version": VERSION, "feature_names": FEATURES, "examples": examples})
    return examples, current, warnings


def _eligible(examples):
    labels = {item["label"] for item in examples}
    return sorted(label for label in labels
                  if sum(item["label"] == label for item in examples) >= 3
                  and len({item["source_identity"] for item in examples if item["label"] == label}) >= 2)


def _predict(features, examples, eligible):
    if features is None:
        return None
    distances = sorted((float(np.linalg.norm(np.array(features)-item["features"])), item["label"])
                       for item in examples if item["label"] in eligible)
    if not distances or distances[0][0] > .8:
        return None
    nearest = distances[:min(3, len(distances))]
    votes = {label: sum(1/(distance+.05) for distance, name in nearest if name == label) for _, label in nearest}
    label, weight = max(votes.items(), key=lambda item: item[1])
    agreement = weight / sum(votes.values())
    if agreement < .67:
        return None
    return {"label": label, "confidence": round(min(.85, agreement / (1+distances[0][0])), 3),
            "method": "local confirmed-example nearest neighbors", "distance": round(distances[0][0], 4)}


def _learning_status(examples, online):
    eligible = _eligible(examples)
    tested, correct, abstained = 0, 0, 0
    # Every held-out example is predicted without any frames from its flight.
    for example in examples:
        if example["label"] not in eligible:
            continue
        training = [item for item in examples if item["source_identity"] != example["source_identity"]]
        prediction = _predict(example["features"], training, eligible)
        tested += 1
        if prediction is None:
            abstained += 1
        elif prediction["label"] == example["label"]:
            correct += 1
    return {"examples": len(examples), "ready": bool(eligible), "enabled_labels": eligible,
            "method": "nearest neighbors on normalized interval motion; no new internet FPV trick training",
            "validation": {"method": "leave-one-flight-out; held-out flight never in neighbors",
                           "examples": tested, "accuracy_including_abstentions": round(correct/tested, 4) if tested else None,
                           "abstentions": abstained},
            "online_model": online,
            "message": online["message"] + (f" Local labels enabled: {', '.join(eligible)}." if eligible else " Local labels require 3 confirmed examples per label from at least 2 source identities.")}


def _scene_interval(samples, start, end):
    chosen = [sample for sample in samples if start <= sample["t"] < end]
    if not chosen:
        return None
    groups = {name: float(np.mean([sample.get("groups", {}).get(name, 0) for sample in chosen]))
              for sample in chosen for name in sample.get("groups", {})}
    if not groups:
        return None
    name, score = max(groups.items(), key=lambda item: item[1])
    return {"label": name if score >= .35 else "uncertain scene", "score": round(score, 3), "groups": groups}


def _motion_event(rows, start, end):
    features = _features(rows, start, end)
    if features is None:
        return "unmeasured interval", 0, "No usable motion samples"
    motion, rotation, dx, dy, proximity, parallax, idle, burst = features
    if idle >= .7:
        return "low-motion interval", .7, "Persistent low proxy motion; hovering and ground footage can both match"
    if burst >= .15 or rotation >= .6:
        return "rotation burst estimate", .65, "Large image rotation; roll, flip, bank, and tracking orbit remain ambiguous"
    if proximity >= .5 and parallax >= .45:
        return "close-pass / weave estimate", .6, "Residual foreground motion after global camera-motion fitting; not proof of a tree gap"
    if motion >= .1:
        return "moving flight line estimate", .5, "Sustained image motion without a dominant rotation burst"
    return "uncertain motion", .25, "Weak or mixed image-motion evidence"


def build_flight_map(analyses, labels, cache, event=lambda *args: None, checkpoint=lambda: None):
    """Build JSON map, attach cached scene samples, and update confirmed examples."""
    checkpoint()
    online = _scene_samples(analyses, cache, event, checkpoint)
    learning_dir = Path(cache) / "learning"
    examples, current, warnings = _dataset(analyses, labels or [], learning_dir)
    learning = _learning_status(examples, online)
    learning["warnings"] = warnings
    if warnings:
        learning["message"] += " " + " ".join(warnings)
    sources = []
    for analysis in analyses:
        checkpoint()
        events = []
        for start in np.arange(0, analysis["duration"], 3):
            end = min(float(start)+3, analysis["duration"])
            label, confidence, reason = _motion_event(analysis["rows"], start, end)
            scene = _scene_interval(analysis.get("scene_samples", []), start, end)
            event_row = {"start": float(start), "end": float(end), "label": label,
                         "confidence": confidence, "method": "motion heuristic", "reason": reason}
            if scene:
                event_row["scene"] = scene
                event_row["reason"] += f"; online scene context: {scene['label']} (score {scene['score']:.2f}, uncalibrated)"
            predicted = _predict(_features(analysis["rows"], start, end), examples, learning["enabled_labels"])
            if predicted:
                event_row["learned_estimate"] = predicted
                event_row["reason"] += f"; local example match: {predicted['label']} (estimate)"
            events.append(event_row)
        for confirmed in current:
            if confirmed["source"] == analysis["source"]:
                events.append({key: confirmed[key] for key in ("start", "end", "label", "confidence", "method")})
                events[-1]["reason"] = "Explicit user-confirmed interval; not inferred by the scene model"
        sources.append({"source": analysis["source"], "identity": analysis["identity"], "duration": analysis["duration"],
                        "events": sorted(events, key=lambda row: (row["start"], row["method"])),
                        "scene_samples": analysis.get("scene_samples", [])})
    result = {"version": VERSION, "sources": sources, "learning": learning,
              "limitations": "Image-motion map, not a geographic route or calibrated named-trick classifier. Scene scores describe sampled surroundings; rapid passes between samples may be missed."}
    _save(learning_dir / "latest-flight-map.json", result)
    return result


def flight_map(analyses, labels, learning_dir):
    return build_flight_map(analyses, labels, Path(learning_dir).parent)


def annotate_candidates(candidates, analyses, learning_dir):
    """Add review context without changing ranking, boundaries, or eligibility."""
    source_map = {analysis["source"]: analysis for analysis in analyses}
    examples = _load_examples(Path(learning_dir) / "confirmed-examples.json")
    eligible = _eligible(examples)
    for candidate in candidates:
        analysis = source_map.get(candidate.get("source"))
        if analysis is None:
            continue
        start, end = candidate["start"], candidate["end"]
        label, confidence, reason = _motion_event(analysis["rows"], start, end)
        method = "motion heuristic"
        # Exact source/time identity is the only route to a confirmed label.
        exact = next((example for example in examples if example["source_identity"] == analysis["identity"]
                      and abs(example["start"]-start) < 1e-4 and abs(example["end"]-end) < 1e-4), None)
        prediction = _predict(_features(analysis["rows"], start, end), examples, eligible)
        if exact:
            label, confidence, method = exact["label"], 1, "user-confirmed"
        elif prediction:
            label, confidence, method = prediction["label"], prediction["confidence"], prediction["method"]
        candidate.update({"flight_label": label, "flight_confidence": confidence, "flight_method": method,
                          "flight_reason": reason})
        scene = _scene_interval(analysis.get("scene_samples", []), start, end)
        if scene:
            candidate["scene_context"] = {**scene, "method": "online-pretrained Places365 scene estimate"}
    return candidates
