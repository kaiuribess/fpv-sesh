"""Maintainer tool: refresh exact wheel hashes from PyPI, then run release tests."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "requirements-lock.txt": {
        "click": "8.3.3", "colorama": "0.4.6", "numpy": "2.2.6",
        "opencv-python-headless": "4.12.0.88", "packaging": "26.3",
        "pillow": "12.3.0", "platformdirs": "4.11.7",
        "scenedetect-headless": "0.7.1", "tqdm": "4.70.0",
    },
    "requirements-dev-lock.txt": {
        "iniconfig": "2.3.0", "pluggy": "1.6.0", "Pygments": "2.21.0", "pytest": "9.0.3",
    },
    "requirements-bootstrap-lock.txt": {"pip": "26.2"},
}


def main():
    packages = []
    for filename, pins in GROUPS.items():
        lines = ["# Exact wheels from PyPI; Windows x64 CPython 3.12 and 3.13.",
                 "# Regenerate with scripts/refresh_core_lock.py; review and test every update."]
        for name, version in pins.items():
            with urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:
                data = json.load(response)
            if data.get("vulnerabilities") or data["info"].get("yanked"):
                raise RuntimeError(f"Review published advisories before pinning {name} {version}")
            wheels = [item for item in data["urls"] if item["filename"].endswith(".whl")
                      and ("-none-any.whl" in item["filename"] or
                           ("win_amd64.whl" in item["filename"] and
                            ("cp312" in item["filename"] or "cp313" in item["filename"]
                             or "abi3" in item["filename"])))]
            if not wheels:
                raise RuntimeError(f"No supported wheel for {name} {version}")
            hashes = sorted({item["digests"]["sha256"] for item in wheels})
            lines.append(f"{name}=={version} " + " ".join(f"--hash=sha256:{digest}" for digest in hashes))
            info = data["info"]
            packages.append({"name": name, "version": version, "group": filename,
                             "source": f"https://pypi.org/project/{name}/{version}/",
                             "license": info.get("license_expression") or info.get("license"),
                             "project_urls": info.get("project_urls"),
                             "published_distributions": [{"filename": item["filename"], "url": item["url"],
                                "sha256": item["digests"]["sha256"]} for item in wheels]})
        (ROOT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (ROOT / "tools/python-dependencies.json").write_text(json.dumps({
        "python": "Windows x64 CPython 3.12 or 3.13; Tkinter required",
        "packages": packages}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Recorded {len(packages)} exact package versions and supported wheel hashes.")


if __name__ == "__main__":
    main()
