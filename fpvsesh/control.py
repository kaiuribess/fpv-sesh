"""Cooperative controls shared by the desktop job and isolated frame workers."""
from __future__ import annotations

import json
from pathlib import Path
import time


class Cancelled(Exception):
    """The user stopped work at a boundary that preserves completed segments."""


def check_control(path: Path, *, on_pause=None, on_resume=None, poll_interval=.1):
    """Wait while paused, or raise between complete frames when cancelled.

    Commands are left in place: deleting a consumed command could remove a newer
    command written by the UI. The parent clears stale controls on job startup.
    A partially written command cannot release an already paused worker.
    """
    if poll_interval <= 0:
        raise ValueError("Control polling interval must be positive")
    paused = False
    while True:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            action = data.get("action") if isinstance(data, dict) else None
        except FileNotFoundError:
            action = "resume"
        except (OSError, ValueError):
            if not paused:
                return
            time.sleep(poll_interval)
            continue
        if action == "cancel":
            raise Cancelled("Cancelled after the current frame; completed segments are cached")
        if action != "pause":
            if paused and on_resume:
                on_resume()
            return
        if not paused:
            paused = True
            if on_pause:
                on_pause()
        time.sleep(poll_interval)
