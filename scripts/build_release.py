"""Build an auditable source ZIP from a clean commit, never local working files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {"input", "music", "output", "cache", "logs", ".git", "sources", "dist"}
MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".m2ts", ".webm", ".wmv", ".3gp",
                  ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".srt",
                  ".safetensors", ".pth", ".pt", ".onnx", ".gguf", ".bin", ".exe", ".dll",
                  ".zip", ".7z", ".whl", ".nupkg", ".msi", ".msix", ".gz"}


def safe_release_path(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts and
            not any(part.lower() in PRIVATE_DIRS or part.lower().startswith(".venv") for part in path.parts) and
            path.suffix.lower() not in MEDIA_SUFFIXES and
            path.name.lower() not in {"status.md", ".env"} and not path.name.lower().startswith(".env.") and
            path.suffix.lower() not in {".pem", ".key"})


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    if git("status", "--porcelain").strip():
        raise SystemExit("Commit the reviewed source changes before building a release.")
    commit = git("rev-parse", "HEAD").decode().strip()
    names = git("ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0")
    names = sorted(name for name in names if name)
    rejected = [name for name in names if not safe_release_path(name)]
    if rejected:
        raise SystemExit("Release contains forbidden local artifacts: " + ", ".join(rejected))
    version_text = git("show", f"{commit}:fpvsesh/__init__.py").decode()
    version = re.search(r'__version__\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', version_text).group(1)
    prefix = f"FPV-Sesh-{version}"
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"{prefix}-Windows-source.zip"
    manifest = {"format_version": 1, "application": "FPV Sesh", "version": version,
                "commit": commit, "distribution": "source; dependencies and optional weights are downloaded separately", "files": {}}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for name in names:
            content = git("show", f"{commit}:{name}")
            manifest["files"][name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            info = zipfile.ZipInfo(f"{prefix}/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, content)
    manifest_path = args.output / f"{prefix}-source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    files = (archive, manifest_path)
    checksums = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in files)
    (args.output / "SHA256SUMS.txt").write_text(checksums, encoding="ascii", newline="\n")
    print(f"Built {archive.name}: {len(names)} source files at {commit}")
    print(checksums, end="")


if __name__ == "__main__":
    main()
