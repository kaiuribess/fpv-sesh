"""Lightweight, local readiness checks with an intentionally path-free report."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import uuid

from . import __version__
from .toolchain import bundled_tool

ROOT = Path(__file__).resolve().parents[1]


def _read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def _path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Expected a relative application asset")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Application asset escapes its folder")
    return path


def _execute(command):
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _runtime():
    version = ".".join(map(str, sys.version_info[:3]))
    compatible = sys.version_info[:2] in ((3, 12), (3, 13)) and struct.calcsize("P") * 8 == 64
    try:
        tkinter = importlib.import_module("tkinter")
        tk_version = str(tkinter.TkVersion)
        tk_available = True
    except (ImportError, OSError):
        tk_version, tk_available = None, False
    return {"python": version, "bits": struct.calcsize("P") * 8, "supported": compatible,
            "tk_available": tk_available, "tk_version": tk_version, "windows": platform.system() == "Windows"}


def _packages(root):
    try:
        lines = (root / "requirements-lock.txt").read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return [], False
    packages = []
    for line in lines:
        match = re.match(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9.+!_-]+)(?:\s|$)", line.strip())
        if not match:
            continue
        name, expected = match.groups()
        try:
            installed = metadata.version(name)
            if not re.fullmatch(r"[A-Za-z0-9.+!_-]{1,80}", installed):
                installed = "unrecognized version"
        except metadata.PackageNotFoundError:
            installed = None
        packages.append({"name": name, "expected": expected, "installed": installed,
                         "matches_lock": installed == expected})
    return packages, bool(packages) and all(item["matches_lock"] for item in packages)


def _tools(root):
    manifest = _read(root / "tools/dependencies.json", {})
    records = manifest.get("tools", []) if isinstance(manifest, dict) else []
    record = next((item for item in records if isinstance(item, dict) and item.get("name") == "FFmpeg"), {}) if isinstance(records, list) else {}
    expected = re.match(r"(\d+\.\d+(?:\.\d+)?)", str(record.get("version", "")))
    expected = expected.group(1) if expected else None
    results = []
    for name, key, required in (("ffmpeg", "executable", True), ("ffprobe", "probe_executable", True),
                                ("ffplay", "player_executable", False)):
        item = {"name": name, "required": required, "present": False, "version": None,
                "integrity_verified": False if required else None, "matches_manifest_version": False if required else None}
        try:
            path = _path(root, record.get(key))
            if not path.is_relative_to((root / "tools").resolve()):
                raise ValueError("Expected a bundled tool")
            item["present"] = path.is_file() and path.stat().st_size > 0
            if item["present"] and required:
                path = bundled_tool(name, root)
                item["integrity_verified"] = True
                result = _execute([str(path), "-version"])
                match = re.search(r"^" + name + r" version (\d+\.\d+(?:\.\d+)?)", result.stdout)
                item["version"] = match.group(1) if result.returncode == 0 and match else None
                item["matches_manifest_version"] = bool(expected and item["version"] == expected)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            pass
        results.append(item)
    return results, all(item["matches_manifest_version"] for item in results if item["required"])


def _gpu():
    devices = []
    try:
        result = _execute(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
        if result.returncode == 0:
            for row in csv.reader(io.StringIO(result.stdout)):
                if len(row) != 3:
                    continue
                name, driver, memory = (value.strip() for value in row)
                if (not re.fullmatch(r"(?:NVIDIA|GeForce|Quadro|Tesla|RTX|GTX|GRID|NVS)[A-Za-z0-9 ()_.-]{1,100}", name) or
                        not re.fullmatch(r"\d+(?:\.\d+){1,3}", driver) or not memory.isdigit()):
                    continue
                devices.append({"name": name, "driver": driver, "memory_mib": int(memory)})
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return {"nvidia_devices": devices, "encode_status": "not benchmarked by doctor",
            "message": "GPU presence does not prove encoder compatibility. Processing tests supported encoders and can use CPU fallback."}


def _optional(root):
    runtime = (root / ".venv-ai/Scripts/python.exe").is_file()
    models = []
    for name, relative in (("Video understanding", "models/qwen3-vl-2b"), ("Scene context", "models/places365")):
        folder = root / relative
        manifest = _read(folder / "manifest.json", {})
        assets = manifest.get("assets", []) if isinstance(manifest, dict) else []
        present = isinstance(assets, list) and bool(assets)
        try:
            present = present and all(_path(folder, asset["file"]).is_file() and
                _path(folder, asset["file"]).stat().st_size == asset["size_bytes"] for asset in assets)
        except (OSError, ValueError, KeyError, TypeError):
            present = False
        models.append({"name": name, "files_present": bool(present), "runtime_present": runtime,
                       "integrity": "not checked by doctor", "inference": "not run"})
    models.append({"name": "AI detail", "files_present": (root / "models/real-esrgan-cuda/RealESRGAN_x2plus.pth").is_file(),
                   "runtime_present": runtime, "integrity": "not checked by doctor", "inference": "not run",
                   "requires_local_sample_validation": True})
    return models


def collect_report(app_dir=ROOT):
    """Read installation state without reading recordings, weights, logs or jobs."""
    root = Path(app_dir).resolve()
    runtime = _runtime()
    packages, packages_ready = _packages(root)
    tools, tools_ready = _tools(root)
    environment = (root / ".venv/Scripts/python.exe").is_file() and Path(sys.prefix).resolve() == root / ".venv"
    runtime["project_environment"] = environment
    checks = [
        {"id": "windows", "required": True, "passed": runtime["windows"], "message": "Windows is required for the desktop release."},
        {"id": "python", "required": True, "passed": runtime["supported"], "message": "Use 64-bit Python 3.12 or 3.13 with a current maintenance release."},
        {"id": "tk", "required": True, "passed": runtime["tk_available"], "message": "Tk is required for the desktop interface."},
        {"id": "environment", "required": True, "passed": environment, "message": "Run install.cmd to prepare the application environment."},
        {"id": "packages", "required": True, "passed": packages_ready, "message": "Core packages must match the pinned installation; rerun install.cmd if needed."},
        {"id": "tools", "required": True, "passed": tools_ready, "message": "Bundled FFmpeg and FFprobe must pass integrity/version checks for the maintained release; rerun install.cmd if needed."},
    ]
    try:
        free_gib = round(shutil.disk_usage(root).free / 1024**3, 2)
    except OSError:
        free_gib = None
    ready = all(item["passed"] for item in checks if item["required"])
    return {"report_kind": "fpv-sesh-readiness", "schema_version": 1, "app_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(), "ready": ready, "checks": checks,
            "runtime": runtime, "packages": packages, "tools": tools, "gpu": _gpu(), "optional_models": _optional(root),
            "free_space_gib": free_gib,
            "privacy": "No usernames, computer names, device UUIDs, absolute paths, footage, job data or environment values are included.",
            "limitations": "Executable integrity/version checks and model file-presence checks only. No downloads, model hashing, inference, encoder benchmark, media decode or render. Processing verifies model assets and estimates job-specific disk space."}


def save_report(path, report):
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Choose a .json output file for the readiness report.")
    if path.exists():
        existing = _read(path, {})
        if not isinstance(existing, dict) or existing.get("report_kind") != "fpv-sesh-readiness":
            raise ValueError("Refusing to replace a non-readiness JSON file; choose a different output file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".partial")
    try:
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check local FPV Sesh readiness without loading models or sharing personal paths.")
    parser.add_argument("--json", action="store_true", help="Print the shareable JSON report")
    parser.add_argument("--output", type=Path, help="Save a shareable JSON readiness report")
    args = parser.parse_args(argv)
    report = collect_report()
    if args.output:
        try:
            save_report(args.output, report)
        except (OSError, ValueError, AttributeError):
            print("Could not save the readiness report. Choose a writable .json file that does not contain other data.", file=sys.stderr)
            return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    else:
        print(f"FPV Sesh {__version__} — {'core editing is ready' if report['ready'] else 'installation needs attention'}")
        for check in report["checks"]:
            print(f"{'OK' if check['passed'] else 'FIX'}  {check['id']}: {check['message']}")
        for model in report["optional_models"]:
            found = model["files_present"] and model["runtime_present"]
            print(f"INFO  {model['name']}: {'files present; checked before use' if found else 'optional files/runtime missing; basic editing still works'}")
        print("INFO  GPU encoding has not been benchmarked by this report; CPU fallback remains available.")
        print("Next: launch.cmd to open the app, or install.cmd to repair required components.")
        if args.output:
            print("Shareable readiness JSON saved. No personal paths or recording details are included.")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
