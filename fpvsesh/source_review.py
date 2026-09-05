"""Interactive source-section playback; no rendering, media writes, or GPU jobs."""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess

from .media import probe
from .toolchain import bundled_tool

ROOT = Path(__file__).resolve().parents[1]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def find_player(app_dir=None):
    """Use only the verified player bundled with this application's FFmpeg."""
    try:
        return bundled_tool("ffplay", app_dir or ROOT)
    except FileNotFoundError as error:
        raise FileNotFoundError("The bundled source player is missing or invalid. Run install.cmd to watch sections.") from error


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
