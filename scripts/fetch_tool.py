"""Fetch and extract the pinned video archive using literal filesystem paths."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.parse import urlparse
from urllib.request import urlopen
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def local_path(relative):
    path = (ROOT / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(ROOT / "tools"):
        raise ValueError("Tool paths must stay inside the application tools folder")
    return path


def matches(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = json.loads((ROOT / "tools/dependencies.json").read_text(encoding="utf-8-sig"))
    record = next(item for item in manifest["tools"] if item["name"] == "FFmpeg" and item["required"])
    url = urlparse(record["url"])
    if url.scheme != "https" or url.netloc != "github.com" or not url.path.startswith("/GyanD/codexffmpeg/releases/download/"):
        raise ValueError("Unexpected release download origin")
    archive, destination = local_path(record["archive"]), local_path(record["extract_to"])
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not matches(archive, record["sha256"]):
        partial = archive.with_name(archive.name + "." + uuid.uuid4().hex + ".download")
        try:
            for attempt in range(3):
                try:
                    with urlopen(record["url"], timeout=60) as response, partial.open("wb") as stream:
                        shutil.copyfileobj(response, stream, 1024 * 1024)
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(2)
            if not matches(partial, record["sha256"]):
                raise RuntimeError("Download checksum mismatch; archive was not used")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    if record["format"] != "zip":
        raise ValueError("The current release requires a ZIP archive")
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            name = PurePosixPath(item.filename)
            target = (destination / item.filename).resolve()
            if name.is_absolute() or ".." in name.parts or not target.is_relative_to(destination) or ":" in item.filename:
                raise ValueError("Unsafe archive member")
        source.extractall(destination)
    for name, field in (("ffmpeg", "executable"), ("ffprobe", "probe_executable"), ("ffplay", "player_executable")):
        if not matches(local_path(record[field]), record["executable_sha256"][name]):
            raise RuntimeError("Extracted tool checksum mismatch")
    print("Published archive and all three installed video-tool checksums verified.")


if __name__ == "__main__":
    main()
