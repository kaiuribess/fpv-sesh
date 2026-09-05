"""Resolve the release's verified tools without discovering unrelated downloads."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def tool_record(root=None):
    root = Path(root or ROOT).resolve()
    try:
        manifest = json.loads((root / "tools/dependencies.json").read_text(encoding="utf-8-sig"))
        return next(item for item in manifest["tools"] if item["name"] == "FFmpeg" and item["required"])
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        raise FileNotFoundError("The video-tool manifest is missing or invalid. Extract a fresh release copy.") from error


@lru_cache(maxsize=16)
def _verified(path, size, modified_ns, changed_ns, expected):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise RuntimeError("A bundled video tool failed its integrity check. Run install.cmd to repair it.")
    return Path(path)


def bundled_tool(kind, root=None):
    root = Path(root or ROOT).resolve()
    record = tool_record(root)
    field = {"ffmpeg": "executable", "ffprobe": "probe_executable", "ffplay": "player_executable"}[kind]
    try:
        relative = record[field]
        expected = record["executable_sha256"][kind]
        if not isinstance(relative, str) or Path(relative).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid path or checksum")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root / "tools") or not path.is_file():
            raise ValueError("Tool must be inside the application tools folder")
        stat = path.stat()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FileNotFoundError(f"The bundled {kind} is missing or invalid. Run install.cmd.") from error
    result = _verified(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError("A video tool changed while it was being checked. Run install.cmd.")
    return result
