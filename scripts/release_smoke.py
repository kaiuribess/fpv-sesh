"""Exercise an isolated synthetic edit, local music and all social previews.

Run with the installed application Python. This needs no GPU or optional model,
never reads user footage, and removes its temporary media and jobs afterward.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpvsesh import media


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exercise():
    started = time.monotonic()
    checks = []

    def require(condition, name):
        checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            raise RuntimeError(name)

    ffmpeg, ffprobe = media.locate_tools()
    with tempfile.TemporaryDirectory(prefix="fpvsesh-release-smoke-") as directory:
        folder = Path(directory).resolve()
        app = folder / "application"
        app.mkdir()
        shutil.copytree(ROOT / "fpvsesh", app / "fpvsesh", ignore=shutil.ignore_patterns("__pycache__"))
        job = app / "output" / "synthetic-review"
        job.mkdir(parents=True)
        sources = []
        rates = ("60/1", "60000/1001")
        for index, rate in enumerate(rates):
            source = folder / f"synthetic café's [flight {index}] & sample.mp4"
            command = [ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                       f"testsrc2=size=320x240:rate={rate}"]
            if index == 0:
                command += ["-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000", "-t", "1.6"]
            command += ["-frames:v", "96", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-vf", "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
                        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
            command += ["-c:a", "aac"] if index == 0 else ["-an"]
            media.run(command + [str(source)], timeout=60)
            sources.append(source)
        music = folder / "synthetic soundtrack.wav"
        media.run([ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                   "sine=frequency=660:sample_rate=48000", "-t", "1.25", "-c:a", "pcm_s16le", str(music)], timeout=60)
        originals = {path: media.sha256_file(path) for path in [*sources, music]}
        # Exact reviewed ranges make the smoke deterministic without pretending
        # that test-chart movement validates automatic trick identification.
        reviews = [{"key": f"synthetic-{index}", "source": str(source), "start": .2, "end": 1.2,
                    "keep": True, "reason": "Synthetic release verification"} for index, source in enumerate(sources)]
        (job / "reviewed-intervals.json").write_text(json.dumps(reviews), encoding="utf-8")
        env = {**os.environ, "FPVSESH_FFMPEG": ffmpeg, "FPVSESH_FFPROBE": ffprobe, "PYTHONIOENCODING": "utf-8"}
        env.pop("PYTHONPATH", None)

        def run_job(arguments):
            result = subprocess.run([sys.executable, "-m", "fpvsesh.cli", "make", "--job", str(job),
                                     "--preview-only", "--recognition", "off", *arguments],
                                    cwd=app, env=env, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                errors = []
                for line in result.stdout.splitlines():
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict) and item.get("stage") == "error":
                            errors.append(str(item.get("message", "Unknown CLI error")))
                    except ValueError:
                        continue
                detail = (errors[-1] if errors else result.stderr[-2400:] or "The CLI exited without a completion event")
                for path, replacement in ((folder, "<synthetic>"), (ROOT, "<application>")):
                    detail = detail.replace(str(path), replacement).replace(path.as_posix(), replacement)
                raise RuntimeError("Synthetic CLI operation failed: " + detail)
            require(_read(job / "status.json")["stage"] == "complete", "CLI completed the requested operation")

        run_job(["--input", str(sources[0]), "--input", str(sources[1]), "--music", str(music),
                 "--music-offset", ".15", "--music-end", "loop", "--quality", "lanczos",
                 "--social-formats", "vertical,square,portrait", "--framing", "blur", "--edit-order", "chronological"])
        timeline = _read(job / "timeline.json")
        require(len(timeline["shots"]) == 2, "Both reviewed source intervals were selected once")
        for index, shot in enumerate(timeline["shots"]):
            require(shot["source"] == str(sources[index]), "Reviewed source order was preserved")
            require(shot["source_start_frame"] == round(.2 * Fraction(rates[index])) and
                    shot["source_end_frame_exclusive"] == round(1.2 * Fraction(rates[index])),
                    "Reviewed approach and exit boundaries were preserved at source-frame precision")
        require(bool(timeline["fps_decision"]["conversions"]), "Mixed frame rates were mapped without changing playback speed")
        require(_read(job / "music-mix.json")["looped"], "Short local music used the requested repeat policy")
        output_sizes = {"preview.mp4": (1280, 720), "social-preview/vertical.mp4": (360, 640),
                        "social-preview/square.mp4": (360, 360), "social-preview/portrait.mp4": (360, 450)}
        for name, (width, height) in output_sizes.items():
            checked = media.validate_output(job / name, timeline["frames"], timeline["fps"], width, height)
            require(checked["passed"] and checked["probe"]["audio"], f"Full decode, timing and audio verified for {name}")
        fingerprint = _read(job / "artifact-state.json")["fingerprint"]
        run_job([])
        require(_read(job / "artifact-state.json")["fingerprint"] == fingerprint and
                not (job / "preview.previous.mp4").exists(), "Unchanged resume kept the saved edit identity and active export")
        run_job(["--no-music", "--social-formats", "none"])
        require(_read(job / "settings.json")["music"] is None and not (job / "music-mix.json").exists(),
                "Removing the music file cleared the saved soundtrack settings")
        for before, after in zip(timeline["shots"], _read(job / "timeline.json")["shots"]):
            require(all(before[key] == after[key] for key in ("source", "start", "end", "frames")),
                    "Music removal preserved the reviewed video timing")
        require(all(media.sha256_file(path) == digest for path, digest in originals.items()),
                "Every original synthetic video and music file retained its full SHA256 identity")
        return {"report_kind": "fpv-sesh-release-smoke", "passed": True,
                "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.monotonic() - started, 2),
                "python": sys.version.split()[0], "ffmpeg_version": media.run([ffmpeg, "-version"], timeout=20).stdout.splitlines()[0],
                "scope": "Synthetic CPU previews and music/social workflow; no user footage, optional models, or GPU inference.",
                "frames": timeline["frames"], "fps": timeline["fps"], "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "logs/release-smoke.json")
    args = parser.parse_args(argv)
    if args.report.exists() and _read(args.report).get("report_kind") != "fpv-sesh-release-smoke":
        parser.error("The report path already contains another kind of document")
    try:
        report = exercise()
    except Exception as error:
        report = {"report_kind": "fpv-sesh-release-smoke", "passed": False,
                  "created_utc": datetime.now(timezone.utc).isoformat(), "error": str(error)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
