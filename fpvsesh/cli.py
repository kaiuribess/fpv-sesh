from __future__ import annotations
import argparse
import hashlib
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from .analysis import analyze, candidates_from_analysis, save_json
from .media import probe, inspect_timestamps, hardware_diagnostics, fps_decision, choose_fps
from .planner import plan
from .render import render_timeline, make_audio, ensure_space
from .control import Cancelled

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"duration": "auto", "style": "hype", "look": "punch", "strength": .55, "quality": "auto", "audio_level": .4, "codec": "hevc"}
MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".m2ts"}

def parser():
    p = argparse.ArgumentParser(description="FPV Sesh — private local multi-clip highlights, no music")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make")
    m.add_argument("--input", action="append")
    m.add_argument("--folder")
    m.add_argument("--job")
    m.add_argument("--duration", choices=["auto", "30", "60", "90", "120"])
    m.add_argument("--style", choices=["hype", "cinematic", "freestyle"])
    m.add_argument("--look", choices=["punch", "natural", "cinematic"])
    m.add_argument("--strength", type=float)
    m.add_argument("--quality", choices=["auto", "lanczos", "ai"])
    m.add_argument("--audio-level", type=float)
    m.add_argument("--codec", choices=["hevc", "h264"])
    m.add_argument("--overrides")
    m.add_argument("--preview-only", action="store_true")
    m.add_argument("--regenerate", action="store_true")
    m.add_argument("--analyze-only", action="store_true")
    sub.add_parser("diagnose")
    return p

def make(args):
    for d in ["input", "music", "output", "cache", "models", "logs"]: (ROOT / d).mkdir(exist_ok=True)
    job = Path(args.job).expanduser().resolve() if args.job else ROOT / "output" / ("Sesh-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    # Jobs and mutable outputs live only inside this application, never input/source directories.
    if not job.is_relative_to(ROOT / "output"): raise ValueError("Job folder must be inside FPV Sesh/output")
    job.mkdir(parents=True, exist_ok=True)
    def event(stage, progress, message):
        entry = {"stage": stage, "progress": round(max(0,min(1,progress)), 4), "message": message, "job": str(job), "time": datetime.now().isoformat()}
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        save_json(job / "status.json", entry)
        with (job / "events.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(entry) + "\n")
    control = ROOT / "cache" / "control.json"
    def checkpoint():
        while control.exists():
            try: action = json.loads(control.read_text(encoding="utf-8")).get("action")
            except (ValueError, OSError): return
            if action == "cancel":
                control.unlink(missing_ok=True)
                raise Cancelled("Cancelled safely at a stage/segment boundary; completed segments are cached")
            if action != "pause":
                control.unlink(missing_ok=True)
                return
            if not getattr(checkpoint, "paused", False):
                event("paused", 0, "Paused at a safe boundary; use Resume to continue")
                checkpoint.paused = True
            time.sleep(.3)
        checkpoint.paused = False
    # OS lock releases automatically even if the process is interrupted.
    lock = (ROOT / "cache" / "run.lock").open("a+b")
    lock.seek(0)
    lock.write(b"0"); lock.flush(); lock.seek(0)
    if os.name == "nt":
        import msvcrt
        try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError: raise RuntimeError("Another FPV Sesh job is running; pause/cancel it before starting another")
    try:
        control.unlink(missing_ok=True)
        saved = json.loads((job / "settings.json").read_text(encoding="utf-8")) if (job / "settings.json").exists() else {}
        settings = {k: getattr(args,k) if getattr(args,k) is not None else saved.get(k,v) for k,v in DEFAULTS.items()}
        if not 0 <= settings["strength"] <= 1 or not 0 <= settings["audio_level"] <= 1: raise ValueError("Strength and audio level must be between 0 and 1")
        paths = args.input or []
        if args.folder:
            folder = Path(args.folder).expanduser().resolve(strict=True)
            paths += [str(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in MEDIA_EXTENSIONS and p.is_file()]
        if not paths: paths = saved.get("inputs", []) or [str(p) for p in sorted((ROOT / "input").glob("*")) if p.suffix.lower() in MEDIA_EXTENSIONS]
        if not paths: raise ValueError("Choose at least one video or place clips in input")
        paths = list(dict.fromkeys(str(Path(p).expanduser().resolve(strict=True)) for p in paths))
        if any(Path(p).is_relative_to(ROOT / "output") or Path(p).is_relative_to(ROOT / "cache") for p in paths): raise ValueError("Use original footage, not FPV Sesh outputs or cache files")
        settings.update({"inputs": paths, "music": None, "stabilization": "disabled: no validated gyro synchronization or calibration"})
        save_json(job / "settings.json", settings)
        event("diagnostics", 0, "Inspecting hardware and source identities")
        ensure_space(ROOT, 5 * 2**30)
        diagnostics = hardware_diagnostics()
        # Backend facts are separate from GPU presence.
        backend_path = ROOT / "logs" / "backend-benchmark.json"
        if backend_path.exists(): diagnostics["backend"] = json.loads(backend_path.read_text(encoding="utf-8"))
        try:
            from .enhance import backend_status
            diagnostics.update(backend_status())
        except (ImportError, AttributeError): pass
        save_json(ROOT / "logs" / "diagnostics.json", diagnostics)
        probes, seen = [], set()
        warnings = []
        for i, path in enumerate(paths):
            checkpoint()
            event("probe", i / len(paths), f"Inspecting {Path(path).name}")
            p = probe(path)
            if p["sha256"] in seen:
                warnings.append(f"Exact duplicate input skipped: {p['filename']}")
                continue
            seen.add(p["sha256"])
            if p["sample_aspect_ratio"] not in ("1:1", "1/1", "unknown", "N/A"):
                raise ValueError(f"{p['filename']}: non-square pixel aspect {p['sample_aspect_ratio']} requires a validated display-aspect normalization path")
            if p["sample_aspect_ratio"] in ("unknown", "N/A"):
                warnings.append(f"{p['filename']}: no pixel-aspect tag; square pixels assumed and the 4:3 picture is preserved when dimensions are 1440x1080")
            if p["hdr"]: raise ValueError(f"HDR input requires a validated tone-map path; this SDR version cannot interpret {p['filename']}")
            for tag in ("color_space", "color_transfer", "color_primaries"):
                if p.get(tag) not in ("bt709", "unknown"): raise ValueError(f"Unsupported color interpretation {tag}={p[tag]} in {p['filename']}; no silent log/HDR conversion")
            if p.get("color_range") not in ("tv", "unknown"): raise ValueError("Full-range input needs a validated range conversion before this SDR pipeline")
            if any(p.get(tag) == "unknown" for tag in ("color_space", "color_transfer", "color_primaries", "color_range")):
                warnings.append(f"{p['filename']}: missing color metadata assumed SDR Rec.709 limited range; review input interpretation")
            timestamp_file = ROOT / "cache" / (p["sha256"][:20] + "-timestamps.json")
            if timestamp_file.exists(): scan = json.loads(timestamp_file.read_text(encoding="utf-8"))
            else:
                event("source-validation", i / len(paths), f"Checking every source timestamp: {p['filename']}")
                scan = inspect_timestamps(path, include_pts=True)
                save_json(timestamp_file, scan)
            if not scan["scan_complete"] or not scan["strictly_monotonic"] or scan["decode_error_count"]:
                raise ValueError(f"Source has decode/timestamp problems: {p['filename']}; inspect {timestamp_file}")
            p["timestamp_scan"] = {k:v for k,v in scan.items() if k != "pts"}
            p["frame_count"] = scan["frame_count"]
            if scan["vfr"]:
                p["frame_pts"] = scan["pts"]
                warnings.append(f"{p['filename']}: VFR mapped by actual presentation timestamps; output uses explicit constant-rate resampling")
            probes.append(p)
        save_json(job / "sources.json", probes)
        analyses = []
        for p in probes:
            checkpoint()
            analyses.append(analyze(p, ROOT / "cache", event, checkpoint))
        # Split recordings are only flagged; filename sequence alone never proves continuity.
        save_json(job / "analysis-summary.json", [{k:v for k,v in a.items() if k != "rows"} for a in analyses])
        review_path = job / "reviewed-intervals.json"
        reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else []
        candidates = candidates_from_analysis(analyses, settings["style"], reviews)
        overrides_path = Path(args.overrides) if args.overrides else job / "overrides.json"
        overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
        save_json(job / "overrides.json", overrides)
        fps = choose_fps(probes)
        timeline = plan(candidates, probes, fps, settings["duration"], settings["style"], overrides)
        timeline["fps_decision"] = fps_decision(probes)
        timeline["warnings"].extend(warnings)
        timeline["input_notes"] = "Sequential filenames may be separate flights or split recordings; no continuity inferred from names. Gyro stabilization skipped without validated gyro data/calibration."
        fingerprint = hashlib.sha256(json.dumps({"timeline":timeline,"settings":settings},sort_keys=True).encode()).hexdigest()
        artifact_state = job / "artifact-state.json"
        old = json.loads(artifact_state.read_text(encoding="utf-8")) if artifact_state.exists() else {}
        if old.get("fingerprint") != fingerprint:
            for filename in ("final_4k.mp4", "preview.mp4"):
                original = job / filename
                if original.exists(): original.replace(original.with_name(original.stem + ".previous.mp4"))
        save_json(artifact_state, {"fingerprint":fingerprint,"music":None})
        save_json(job / "timeline.json", timeline)
        save_json(job / "candidates.json", {"candidates": candidates, "coverage": "entire session at 6 motion samples/sec, 12 fps scene detection proxy", "overrides": overrides})
        event("timeline", 1, f"Selected {len(timeline['shots'])} distinct moments, {timeline['duration']:.2f}s from {len(probes)} clips. No music.")
        if args.analyze_only:
            event("complete", 1, "Full-session analysis and reviewable timeline ready")
            return
        audio = make_audio(timeline, probes, job, settings["audio_level"], event, checkpoint)
        if audio is None: (job / "source-audio.m4a").unlink(missing_ok=True)
        preview = render_timeline(timeline, probes, settings, job, ROOT / "cache", event, checkpoint, preview=True)
        final = None
        if not args.preview_only:
            checkpoint()
            final = render_timeline(timeline, probes, settings, job, ROOT / "cache", event, checkpoint)
        # Compare source identities after processing; originals are never written.
        for p in probes:
            stat = Path(p["source"]).stat()
            if stat.st_size != p["size_bytes"] or stat.st_mtime_ns != p["mtime_ns"]: raise RuntimeError(f"Source changed externally during processing: {p['filename']}")
        report = build_report(timeline, settings, probes, diagnostics, preview, final)
        (job / "report.md").write_text(report, encoding="utf-8")
        event("complete", 1, f"Finished {'4K reel and preview' if final else 'preview'}: {job}")
    except Cancelled as e:
        event("cancelled", 0, str(e))
    except Exception as e:
        event("error", 0, str(e))
        raise
    finally:
        lock.close()

def build_report(timeline, settings, probes, diagnostics, preview, final):
    lines = ["# FPV Sesh run report", "", f"{len(timeline['shots'])} distinct shots • {timeline['duration']:.3f} seconds • {timeline['fps']} fps • no music", "",
             f"Style: {settings['style']}; look: {settings['look']} at {settings['strength']:.0%}; original-audio level: {settings['audio_level']:.0%}.",
             "", "Sources were decoded throughout and analyzed across their entire duration. Originals retained unchanged (size/mtime checked after run, complete SHA256 recorded before run).",
             "", "Source intervals and frame mappings are in timeline.json; all candidates and selection/rejection reasons are in candidates.json.",
             "", "## Output", f"Preview: {preview['path']}"]
    if final:
        lines += [f"4K final: {final['path']}", "", "3840×2160 canvas, original aspect ratio contained with side bars when necessary. HEVC Main10 is an output format; it does not recover 10-bit camera information from 8-bit sources.", "", "Backends:"]
        backend_names = sorted({r['backend'] + " + " + r['encoder'] for r in final["records"]})
        lines += ["- " + name for name in backend_names]
        v = final['verification']['probe']
        memory_samples = [r['peak_total_gpu_memory_mib'] for r in final['records'] if r.get('peak_total_gpu_memory_mib') is not None]
        lines += [f"- Verified format: {v['codec']}, {v['profile']}, {v['pix_fmt']}, {v['color_space']}, {v['color_range']} range.",
                  f"- AI inference in final: {any(r.get('ai_inference',False) for r in final['records'])}.",
                  f"- Total selected-interval render time: {sum(r.get('elapsed_seconds',0) for r in final['records']):.2f} seconds.",
                  (f"- Sampled peak total GPU memory: {max(memory_samples)} MiB (includes other GPU users)." if memory_samples else "- Total GPU memory sampling unavailable.")]
        for r in final["records"]:
            lines += ["- Fallback warning: " + w for w in r.get('warnings',[])]
    lines += ["", "## Verification", "Preview: " + str(preview["verification"]["passed"])]
    if final: lines += ["Final: " + str(final["verification"]["passed"])]
    lines += ["Full decode, frame count, rational frame rate, output dimensions, presentation timestamps and black-gap diagnostics saved separately. Original audio is decoded, trimmed, resampled, faded and encoded once, then muxed into both outputs. No invented drone sound. Loudness measurement is saved in audio-loudness.txt when audio exists.",
              "", f"Hardware: {diagnostics.get('gpu_name','see diagnostics')}; {diagnostics.get('vram_mib','unknown')} MiB VRAM; CPU and RAM details in logs/diagnostics.json.",
              "", "## Practical limits", "Selection uses motion, quality and image similarity heuristics. It cannot certify complete tricks or a semantic crash classification. Rotation is never itself treated as a crash. Gyro stabilization, synthetic slow motion, music timing and temporal restoration are disabled. Fixed conservative per-shot color keeps sunset warmth; fully clipped sun detail cannot be recovered. Unknown color tags are explicitly reported. No claim of full-motion human review is made by the application.",
              "", *["- " + w for w in timeline["warnings"]], "", "Installed sources, licenses and benchmarks: application logs/backend-* and dependency records. This is a local editor and no media or metadata is uploaded."]
    return "\n".join(lines) + "\n"

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parser().parse_args()
    if args.command == "diagnose":
        data = hardware_diagnostics()
        try:
            from .enhance import backend_status
            data.update(backend_status())
        except (ImportError, AttributeError): pass
        save_json(ROOT / "logs" / "diagnostics.json", data)
        print(json.dumps(data, indent=2))
    else: make(args)

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(json.dumps({"stage":"error", "progress":0, "message":str(exc)}), flush=True)
        sys.exit(1)
