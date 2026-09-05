"""Checks shared by setup and release verification; no optional model imports."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]


def locked_versions(path):
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s]+)", Path(path).read_text(encoding="utf-8"), re.M))


def check_packages(root=ROOT):
    failures = []
    for name, expected in locked_versions(Path(root) / "requirements-lock.txt").items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "missing"
        if installed != expected:
            failures.append(f"{name}: expected {expected}, found {installed}")
    return failures


def main():
    if sys.version_info[:2] not in {(3, 12), (3, 13)} or struct.calcsize("P") != 8:
        raise SystemExit("Use 64-bit Python 3.12 or 3.13 with Tkinter.")
    import tkinter
    import cv2
    import numpy
    import PIL.Image
    import scenedetect
    failures = check_packages()
    if failures:
        raise SystemExit("\n".join(failures))
    from .media import locate_tools, run
    ffmpeg, ffprobe = locate_tools()
    from .toolchain import bundled_tool
    bundled_tool("ffplay")
    run([ffprobe, "-v", "error", "-version"], timeout=30)
    run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=64x64:rate=10:duration=0.2", "-c:v", "libx264", "-f", "null", "-"], timeout=30)
    print("Required packages, verified video tools and CPU encoding passed.")


if __name__ == "__main__":
    main()
