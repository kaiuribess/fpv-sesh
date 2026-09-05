"""Keep optional Windows DLL discovery inside this interpreter's environment."""
from __future__ import annotations

import os
from pathlib import Path
import sys

_handles = {}


def prepare_torch_dlls():
    """Set this process's search path before importing Torch; no system changes.

    cuDNN loads further libraries by name. Torch's own discovery can miss those
    when its installation folder contains brackets. Resolve a literal directory
    under sys.prefix, never a caller-supplied directory or a different runtime.
    Setup verifies publisher wheel hashes; runtime preflight checks installed
    metadata. This helper only constrains the library search location.
    """
    if sys.platform != "win32":
        return
    prefix = Path(sys.prefix).resolve(strict=True)
    library = (prefix / "Lib/site-packages/torch/lib").resolve(strict=True)
    if not library.is_relative_to(prefix) or not library.is_dir():
        raise RuntimeError("Torch libraries must be inside this Python environment")
    if not all((library / name).is_file() for name in ("torch_cpu.dll", "torch_python.dll")):
        raise RuntimeError("Torch libraries are incomplete; run install-models.cmd to repair them")
    location = str(library)
    if location not in _handles:
        _handles[location] = os.add_dll_directory(location)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([location] + [entry for entry in entries if entry != location])

