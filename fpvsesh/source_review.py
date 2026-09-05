"""Interactive source-section playback; no rendering, media writes, or GPU jobs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from .media import probe

ROOT = Path(__file__).resolve().parents[1]
# This executable was independently hashed both in the installed directory and
# directly from the archive pinned in tools/dependencies.json. No new download.
FFMPEG_ARCHIVE_SHA256 = "e5bbf665b9a43219d97a15c2dd2cdeb9b11bcb689d88a2177c4f754b4e20f793"
FFPLAY_SHA256 = "ebc3c2e4543961f7701d2c9383aef9cebf0357db5fecd9aaec9479ed1a4ce5dc"


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def find_player(app_dir=None):
    """Use only the verified player bundled with this application's FFmpeg."""
    root = Path(app_dir or ROOT).resolve()
    try:
        manifest = json.loads((root / "tools/dependencies.json").read_text(encoding="utf-8-sig"))
        records = manifest.get("tools", []) if isinstance(manifest, dict) else []
        record = next(item for item in records if isinstance(item, dict) and item.get("name") == "FFmpeg"
                      and item.get("sha256") == FFMPEG_ARCHIVE_SHA256)
        executable = record["executable"]
        if not isinstance(executable, str) or Path(executable).is_absolute():
            raise ValueError("Expected a bundled FFmpeg path")
        player = (root / executable).with_name("ffplay.exe").resolve(strict=True)
        if not player.is_relative_to(root / "tools") or not player.is_file():
            raise ValueError("Expected the bundled FFplay executable")
    except (OSError, ValueError, TypeError, KeyError, StopIteration) as error:
        raise FileNotFoundError("The bundled source player is missing. Restore the project's FFmpeg 7.1.1 tools to watch sections.") from error
    with player.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != FFPLAY_SHA256:
        raise RuntimeError("The bundled source player failed its integrity check. Restore the project's verified FFmpeg 7.1.1 tools.")
    return player


def normalize_section(source, start, end, *, source_duration=None, context=2.0):
    """Validate the requested source range and clamp context to actual media."""
    if (not _finite(start) or not _finite(end) or not 0 <= start < end or
            not _finite(context) or not 0 <= context <= 30):
        raise ValueError("Choose a finite source range with 0 <= start < end and context between 0 and 30 seconds")
    if source_duration is not None and (not _finite(source_duration) or source_duration <= 0):
        raise ValueError("Saved source duration must be a finite positive number")
    path = Path(source).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Choose an existing, nonempty video recording")
    # This reads container metadata only: no complete source hash or GPU decode
    # is needed when the user merely opens a review player.
    metadata = probe(path, include_hash=False)
    duration = metadata.get("duration")
    if not _finite(duration) or duration <= 0:
        raise ValueError("The recording has no usable duration")
    if end > duration+1e-6:
        raise ValueError("The selected section extends beyond the current recording; refresh its flight map")
    requested_end = min(float(end), float(duration))
    if start >= requested_end:
        raise ValueError("The selected section contains no source time")
    context_start, context_end = max(0.0, float(start)-context), min(float(duration), requested_end+context)
    return {"source": str(path), "requested_start": float(start), "requested_end": requested_end,
            "source_duration": float(duration), "start": context_start, "end": context_end,
            "duration": context_end-context_start, "context_seconds": float(context)}


def play_section(source, start, end, *, source_duration=None, context=2.0, app_dir=None):
    """Open the selected source interval only when called by a user action.

    Uses software decoding/display with two decoder threads so playback does
    not take the CUDA device used by an active recognition job. No job lock,
    control file, source, cache, timeline, or export is written.
    """
    section = normalize_section(source, start, end, source_duration=source_duration, context=context)
    player = find_player(app_dir)
    title = f"FPV Sesh | {Path(section['source']).name} | {section['start']:.2f}-{section['end']:.2f}s"
    command = [str(player), "-hide_banner", "-loglevel", "error", "-autoexit", "-sn",
               "-ss", f"{section['start']:.6f}", "-t", f"{section['duration']:.6f}",
               "-window_title", title, "-x", "1100", "-y", "720", "-seek_interval", "1",
               "-threads", "2", "-filter_threads", "1", "-i", section["source"]]
    environment = os.environ.copy()
    environment["SDL_RENDER_DRIVER"] = "software"
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, shell=False, env=environment,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as error:
        raise RuntimeError("The source player could not open this section: " + str(error)) from error
    return {**section, "player": str(player), "pid": process.pid}
