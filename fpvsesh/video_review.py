"""Refresh existing-job recognition without touching edit timing or exports."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile

from .analysis import analyze, save_json
from .control import Cancelled, check_control
from .flightmap import annotate_candidates, build_flight_map
from .media import probe

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PREFIXES = ("flight_", "trick_", "scene_")


@contextmanager
def _run_lock(cache):
    cache.mkdir(parents=True, exist_ok=True)
    lock = (cache / "run.lock").open("a+b")
    try:
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another FPV Sesh job is running; finish or cancel it before rescanning") from error
        yield
    finally:
        lock.close()


def _snapshot(path):
    return path.read_bytes() if path.exists() else None


def _read(snapshot, name, expected_type, optional=None):
    if snapshot is None:
        if optional is not None:
            return copy.deepcopy(optional)
        raise ValueError(f"Existing job is missing {name}; analyze the recordings first")
    try:
        value = json.loads(snapshot.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Existing job has invalid {name}") from error
    if not isinstance(value, expected_type):
        raise ValueError(f"Existing job has invalid {name}")
    return value


def _replace_bytes(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".recognition-", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _commit(job, updates, before):
    # Prepare serialization before the first mutation. Failed inference, invalid
    # values, and cancellation therefore cannot replace an existing job map.
    encoded = {name: json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
               for name, value in updates.items()}
    changed = []
    try:
        for name, content in encoded.items():
            _replace_bytes(job / name, content)
            changed.append(name)
    except Exception:
        for name in reversed(changed):
            if before[name] is None:
                (job / name).unlink(missing_ok=True)
            else:
                _replace_bytes(job / name, before[name])
        raise


def map_flight(job, recognition="auto"):
    """Rescan a verified saved job; only recognition annotations may change.

    Candidates keep exact IDs, boundaries, scores, selection, and overrides.
    Timeline, sources, render status, artifact fingerprints, audio, and export
    files are never written. Recognition progress has its own status file.
    """
    if recognition not in {"auto", "off", "thorough"}:
        raise ValueError("Recognition must be auto, off, or thorough")
    root = ROOT.resolve()
    output = (root / "output").resolve()
    job = Path(job).expanduser().resolve(strict=True)
    if (not output.is_relative_to(root) or not job.is_dir() or
            job == output or not job.is_relative_to(output)):
        raise ValueError("Choose an existing job folder inside FPV Sesh/output")
    cache = root / "cache"

    def event(stage, progress, message, **metadata):
        entry = {"stage": stage, "progress": round(max(0, min(1, progress)), 4),
                 "message": message, "job": str(job), "time": datetime.now().isoformat(),
                 "operation": "map-flight", **metadata}
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        save_json(job / "recognition-status.json", entry)

    control = cache / "control.json"

    def checkpoint():
        check_control(control,
                      on_pause=lambda: event("paused", 0, "Flight recognition paused; use Resume to continue"),
                      on_resume=lambda: event("recognition", 0, "Resuming flight recognition"))

    with _run_lock(cache):
        # Clear only a stale command from the previous completed operation.
        control.unlink(missing_ok=True)
        try:
            names = ("sources.json", "settings.json", "candidates.json", "flight-map.json", "flight-labels.json")
            before = {name: _snapshot(job / name) for name in names}
            sources = _read(before["sources.json"], "sources.json", list)
            settings = _read(before["settings.json"], "settings.json", dict)
            package = _read(before["candidates.json"], "candidates.json", dict)
            labels = _read(before["flight-labels.json"], "flight-labels.json", (list, dict), optional=[])
            if not sources or not isinstance(package.get("candidates"), list):
                raise ValueError("The saved job needs analyzed sources and review candidates before rescanning")
            verified = []
            for index, saved in enumerate(sources):
                checkpoint()
                digest = saved.get("sha256") if isinstance(saved, dict) else None
                if (not isinstance(digest, str) or len(digest) != 64 or
                        any(char not in "0123456789abcdef" for char in digest) or
                        not isinstance(saved.get("source"), str)):
                    raise ValueError("Saved sources need complete SHA256 identities before rescanning")
                path = Path(saved["source"]).expanduser().resolve(strict=True)
                if path.is_relative_to(output) or path.is_relative_to(cache.resolve()):
                    raise ValueError("Saved recordings must reference original footage outside output and cache")
                event("source-validation", index / len(sources), f"Checking the complete source identity: {path.name}")
                current = probe(path, include_hash=True)
                if current["sha256"] != digest:
                    raise ValueError(f"Recording changed since this job was analyzed: {path.name}. Existing edits and maps were preserved.")
                duration = saved.get("duration")
                if (isinstance(duration, bool) or not isinstance(duration, (int, float)) or
                        not math.isfinite(duration) or duration <= 0 or
                        abs(duration - current["duration"]) > .001):
                    raise ValueError(f"Saved source duration is inconsistent for {path.name}; analyze a new job")
                verified.append(current)
            identities = {saved["source"]: saved["sha256"] for saved in sources}
            durations = {saved["source"]: saved["duration"] for saved in sources}
            for candidate in package["candidates"]:
                if not isinstance(candidate, dict) or candidate.get("source") not in identities:
                    raise ValueError("A saved candidate references an unknown recording")
                if candidate.get("identity") != identities[candidate["source"]]:
                    raise ValueError("A saved candidate has a stale source identity; analyze a new job")
                start, end = candidate.get("start"), candidate.get("end")
                if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                        for value in (start, end)) or not 0 <= start < end <= durations[candidate["source"]]):
                    raise ValueError("A saved candidate has invalid source boundaries")
            # No analysis or map annotations are written until EVERY original
            # has passed its full hash check against the saved job identities.
            analyses = []
            for saved in sources:
                checkpoint()
                analyses.append(analyze(copy.deepcopy(saved), cache, event, checkpoint))
            result = build_flight_map(analyses, labels, cache, event, checkpoint, recognition=recognition)
            annotated = copy.deepcopy(package["candidates"])
            for candidate in annotated:
                for key in tuple(candidate):
                    if key.startswith(ANNOTATION_PREFIXES):
                        del candidate[key]
            annotate_candidates(annotated, analyses, cache / "learning")
            updated = copy.deepcopy(package)
            for original, replacement, measured in zip(package["candidates"], updated["candidates"], annotated):
                for key in tuple(original):
                    if key.startswith(ANNOTATION_PREFIXES):
                        replacement.pop(key, None)
                replacement.update({key: value for key, value in measured.items() if key.startswith(ANNOTATION_PREFIXES)})
            checkpoint()
            for source in verified:
                stat = Path(source["source"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
                    raise RuntimeError("A recording changed while recognition was running; existing job annotations were preserved")
            for name, content in before.items():
                if _snapshot(job / name) != content:
                    raise RuntimeError(f"{name} changed while recognition was running; rescan again after saving edits")
            settings["recognition"] = recognition
            _commit(job, {"flight-map.json": result, "candidates.json": updated, "settings.json": settings}, before)
            learning = result.get("learning", {})
            video = learning.get("video_model", {}) if isinstance(learning, dict) else {}
            video = video if isinstance(video, dict) else {}
            complete = recognition == "off" or video.get("coverage_complete") is True
            coverage = {"coverage_complete": complete,
                        "windows_analyzed": video.get("windows_analyzed", 0),
                        "windows_requested": video.get("windows_requested", 0),
                        "coverage_seconds": video.get("coverage_seconds", 0)}
            if complete:
                message = ("Video recognition switched off. Motion and scene results are saved; your finished edit is unchanged."
                           if recognition == "off" else
                           "Flight recognition refreshed. Existing edit timing and exported videos are preserved.")
            elif coverage["windows_requested"]:
                message = (f"Flight understanding partly updated: {coverage['windows_analyzed']} of "
                           f"{coverage['windows_requested']} sections are available. Refresh understanding to continue. "
                           "Your finished edit is unchanged.")
            else:
                message = ("Flight understanding partly updated. Video recognition was unavailable; motion and scene "
                           "results are saved. Your finished edit is unchanged.")
            event("complete" if complete else "partial", 1, message, **coverage,
                  details=video.get("message", ""))
            return {"job": str(job), "recognition": recognition, "sources": len(sources),
                    "candidates": len(updated["candidates"]), "cancelled": False, "partial": not complete,
                    **coverage}
        except Cancelled as error:
            event("cancelled", 0, "Flight recognition cancelled. Existing job maps and exported videos are preserved.")
            return {"job": str(job), "cancelled": True, "message": str(error)}
        except Exception as error:
            event("error", 0, str(error))
            raise
