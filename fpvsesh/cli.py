from __future__ import annotations
import argparse
import hashlib
from datetime import datetime
import json
from pathlib import Path
import sys
from .analysis import analyze, candidates_from_analysis, save_json
from .media import probe, inspect_timestamps, hardware_diagnostics, fps_decision, choose_fps
from .planner import plan
from .render import render_timeline, make_audio, ensure_space
from .control import Cancelled, acquire_run_lock, check_control
from .settings import DEFAULTS, resolve_settings

ROOT = Path(__file__).resolve().parents[1]
MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".m2ts"}

def parser():
    p = argparse.ArgumentParser(description="FPV Sesh — local FPV highlights, music and social exports")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make")
    m.add_argument("--input", action="append")
    m.add_argument("--folder")
    m.add_argument("--job")
    m.add_argument("--duration", choices=["auto", "15", "30", "60", "90", "120", "180"])
    m.add_argument("--style", choices=["hype", "cinematic", "freestyle", "flow"])
    m.add_argument("--look", choices=["punch", "natural", "cinematic"])
    m.add_argument("--strength", type=float)
    m.add_argument("--quality", choices=["auto", "lanczos", "ai"])
    m.add_argument("--audio-level", type=float)
    m.add_argument("--codec", choices=["hevc", "h264"])
    music = m.add_mutually_exclusive_group()
    music.add_argument("--music", help="Local music file to mix under the edit")
    music.add_argument("--no-music", action="store_true", help="Remove music, including saved-job music")
    m.add_argument("--music-level", type=float)
    m.add_argument("--music-offset", type=float, help="Start music this many seconds into the track")
    m.add_argument("--music-fade", type=float)
    m.add_argument("--music-end", choices=["fade", "loop"])
    m.add_argument("--beat-sync", action=argparse.BooleanOptionalAction, default=None)
    m.add_argument("--social-formats", help="vertical,square,portrait or none; 4K master is always included")
    m.add_argument("--framing", choices=["blur", "fit", "fill"])
    m.add_argument("--focus-x", type=float)
    m.add_argument("--edit-order", choices=["story", "chronological"])
    m.add_argument("--recovery", type=float, help="Seconds of follow-through after motion bursts (0.5–8)")
    m.add_argument("--recognition", choices=["auto", "off", "thorough"], help="Local online-pretrained video understanding")
    m.add_argument("--overrides")
    m.add_argument("--preview-only", action="store_true")
    m.add_argument("--regenerate", action="store_true")
    m.add_argument("--analyze-only", action="store_true")
    sub.add_parser("diagnose")
    validation = sub.add_parser("validate-ai", help="Render and verify a local CUDA sample before enabling AI")
    validation.add_argument("--input", required=True)
    validation.add_argument("--start", type=float, default=0)
    validation.add_argument("--seconds", type=float, default=2)
    review = sub.add_parser("map-flight", help="Refresh an existing flight map without changing or rendering its edit")
    review.add_argument("--job", required=True)
    review.add_argument("--recognition", choices=["auto", "off", "thorough"], default="auto")
    return p


def merge_reviewed_keeps(overrides, candidates, reviews):
    """Review defaults can add keeps; explicit user exclusions take precedence.

    Existing contradictory explicit keep/exclude choices remain intact so the
    planner reports them instead of silently choosing for the user.
    """
    keys = set()
    if not isinstance(reviews, list):
        raise ValueError("Reviewed ranges must be a list with unique nonempty keys")
    for review in reviews:
        key = review.get("key") if isinstance(review, dict) else None
        if not isinstance(key, str) or not key.strip() or key in keys:
            raise ValueError("Each reviewed range must have a unique nonempty string key; repair or re-create the saved ranges")
        keys.add(key)
    result = dict(overrides)
    reviewed_keep = {review.get("key") for review in reviews if review.get("keep") is True}
    if reviewed_keep:
        excluded = set(result.get("exclude", []))
        result["keep"] = list(dict.fromkeys(result.get("keep", []) +
                              [candidate["id"] for candidate in candidates
                               if candidate.get("review_key") in reviewed_keep and candidate["id"] not in excluded]))
    return result

def make(args):
    for d in ["input", "music", "output", "cache", "models", "logs"]: (ROOT / d).mkdir(exist_ok=True)
    job = Path(args.job).expanduser().resolve() if args.job else ROOT / "output" / ("Sesh-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    # Jobs and mutable outputs live only inside this application, never input/source directories.
    if job == (ROOT / "output").resolve() or not job.is_relative_to((ROOT / "output").resolve()):
        raise ValueError("Choose a job folder inside FPV Sesh/output, not the output folder itself")
    job.mkdir(parents=True, exist_ok=True)
    def event(stage, progress, message):
        entry = {"stage": stage, "progress": round(max(0,min(1,progress)), 4), "message": message, "job": str(job), "time": datetime.now().isoformat()}
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        save_json(job / "status.json", entry)
        with (job / "events.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(entry) + "\n")
    control = ROOT / "cache" / "control.json"
    def checkpoint():
        check_control(control,
                      on_pause=lambda: event("paused", 0, "Paused at a safe boundary; use Resume to continue"),
                      on_resume=lambda: event("resumed", 0, "Resuming the saved operation"))
    # OS lock releases automatically even if the process is interrupted.
    lock = acquire_run_lock(ROOT / "cache")
    try:
        control.unlink(missing_ok=True)
        saved = json.loads((job / "settings.json").read_text(encoding="utf-8")) if (job / "settings.json").exists() else {}
        previous_sources = json.loads((job / "sources.json").read_text(encoding="utf-8")) if (job / "sources.json").exists() else []
        previous_identities = {p["source"]: p["sha256"] for p in previous_sources}
        settings = resolve_settings(args, saved)
        paths = args.input or []
        if args.folder:
            folder = Path(args.folder).expanduser().resolve(strict=True)
            paths += [str(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in MEDIA_EXTENSIONS and p.is_file()]
        if not paths: paths = saved.get("inputs", []) or [str(p) for p in sorted((ROOT / "input").glob("*")) if p.suffix.lower() in MEDIA_EXTENSIONS]
        if not paths: raise ValueError("Choose at least one video or place clips in input")
        paths = list(dict.fromkeys(str(Path(p).expanduser().resolve(strict=True)) for p in paths))
        if any(Path(p).is_relative_to(ROOT / "output") or Path(p).is_relative_to(ROOT / "cache") for p in paths): raise ValueError("Use original footage, not FPV Sesh outputs or cache files")
        music_info = None
        if settings["music"]:
            music_path = Path(settings["music"]).expanduser().resolve(strict=True)
            if music_path.is_relative_to(ROOT / "output") or music_path.is_relative_to(ROOT / "cache"):
                raise ValueError("Choose an original music file outside FPV Sesh/output and cache; copy the track into music/ or another folder first so generated audio cannot overwrite it")
            from .music import analyze_music
            event("music", 0, "Checking the music file and finding clear rhythm accents")
            music_info = analyze_music(settings["music"], ROOT / "cache", settings["music_offset"], checkpoint)
            # Cache diagnostics are not editing decisions and must not retire
            # a completed export when an unchanged soundtrack is reused.
            music_info.pop("cache_hit", None)
            settings["music"] = music_info["path"]
            settings["music_sha256"] = music_info["sha256"]
        settings.update({"inputs": paths, "stabilization": "disabled: no validated gyro synchronization or calibration"})
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
        # A rejected source or cancelled preflight must leave the previously
        # saved edit's settings and soundtrack records intact. Persist resume
        # inputs only after EVERY recording passes interpretation and decoding.
        checkpoint()
        save_json(job / "settings.json", settings)
        if music_info:
            save_json(job / "music-analysis.json", music_info)
        else:
            (job / "music-analysis.json").unlink(missing_ok=True)
            (job / "music-mix.json").unlink(missing_ok=True)
        save_json(job / "sources.json", probes)
        analyses = []
        for p in probes:
            checkpoint()
            analyses.append(analyze(p, ROOT / "cache", event, checkpoint))
        # Split recordings are only flagged; filename sequence alone never proves continuity.
        save_json(job / "analysis-summary.json", [{k:v for k,v in a.items() if k != "rows"} for a in analyses])
        from .flightmap import build_flight_map, annotate_candidates
        labels_path = job / "flight-labels.json"
        labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else []
        flight_map = build_flight_map(analyses, labels, ROOT / "cache", event, checkpoint, recognition=settings["recognition"])
        save_json(job / "flight-map.json", flight_map)
        review_path = job / "reviewed-intervals.json"
        reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else []
        current_identities = {p["source"]: p["sha256"] for p in probes}
        for reviewed in reviews:
            # Migrate saved ranges against the previously verified source,
            # never silently rebind them when a recording is replaced in place.
            if not reviewed.get("source_identity"):
                reviewed["source_identity"] = previous_identities.get(reviewed["source"], current_identities.get(reviewed["source"]))
        if reviews:
            save_json(review_path, reviews)
        candidates = candidates_from_analysis(analyses, settings["style"], reviews, settings["recovery"],
                                             None if settings["duration"] == "auto" else int(settings["duration"]))
        annotate_candidates(candidates, analyses, ROOT / "cache" / "learning")
        overrides_path = Path(args.overrides) if args.overrides else job / "overrides.json"
        overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
        overrides = merge_reviewed_keeps(overrides, candidates, reviews)
        save_json(job / "overrides.json", overrides)
        fps = choose_fps(probes)
        timeline = plan(candidates, probes, fps, settings["duration"], settings["style"], overrides, settings["edit_order"])
        if music_info:
            if settings["beat_sync"] and settings["music_level"] > 0:
                from .pacing import favor_beats
                timeline = favor_beats(timeline, probes, analyses, music_info, settings["recovery"])
            timeline["music"] = music_info
            timeline["music_status"] = f"Local music: {Path(music_info['path']).name}"
            timeline["warnings"].extend(music_info.get("warnings", []))
        timeline["fps_decision"] = fps_decision(probes)
        timeline["warnings"].extend(warnings)
        timeline["input_notes"] = "Sequential filenames may be separate flights or split recordings; no continuity inferred from names. Gyro stabilization skipped without validated gyro data/calibration."
        fingerprint = hashlib.sha256(json.dumps({"timeline":timeline,"settings":settings},sort_keys=True).encode()).hexdigest()
        artifact_state = job / "artifact-state.json"
        old = json.loads(artifact_state.read_text(encoding="utf-8")) if artifact_state.exists() else {}
        if old.get("fingerprint") != fingerprint:
            for filename in ("final_4k.mp4", "preview.mp4", *[f"{folder}/{code}.mp4" for folder in ("social", "social-preview") for code in ("vertical", "square", "portrait")]):
                original = job / filename
                if original.exists(): original.replace(original.with_name(original.stem + ".previous.mp4"))
        save_json(artifact_state, {"fingerprint":fingerprint,"music":settings["music"]})
        save_json(job / "timeline.json", timeline)
        save_json(job / "candidates.json", {"candidates": candidates, "coverage": "entire session at 6 motion samples/sec, 12 fps scene detection proxy", "overrides": overrides})
        from .edit_package import write_edit_package
        write_edit_package(timeline, job)
        event("timeline", 1, f"Selected {len(timeline['shots'])} distinct moments, {timeline['duration']:.2f}s from {len(probes)} clips. {timeline['music_status']}.")
        if args.analyze_only:
            event("complete", 1, "Full-session analysis and reviewable timeline ready")
            return
        audio = make_audio(timeline, probes, job, settings["audio_level"], event, checkpoint, lossless=bool(music_info))
        if audio is None: (job / "source-audio.m4a").unlink(missing_ok=True)
        if music_info:
            from .music import mix_music
            event("audio", 0, "Mixing music with flight sound and fading the edit")
            audio = mix_music(audio, music_info, job, timeline["duration"], settings["music_level"],
                              settings["music_fade"], settings["music_end"], checkpoint)
        preview = render_timeline(timeline, probes, settings, job, ROOT / "cache", event, checkpoint, preview=True)
        from .social import export_social
        social_preview = export_social(timeline, probes, settings, job, ROOT / "cache", event, checkpoint, preview=True) if settings["social_formats"] else {}
        final = None
        social_final = {}
        if not args.preview_only:
            checkpoint()
            final = render_timeline(timeline, probes, settings, job, ROOT / "cache", event, checkpoint)
            social_final = export_social(timeline, probes, settings, job, ROOT / "cache", event, checkpoint) if settings["social_formats"] else {}
        save_json(job / "exports.json", {"preview": preview["path"], "master": final["path"] if final else None,
                                         "social_previews": social_preview, "social": social_final})
        # Compare source identities after processing; originals are never written.
        for p in probes:
            stat = Path(p["source"]).stat()
            if stat.st_size != p["size_bytes"] or stat.st_mtime_ns != p["mtime_ns"]: raise RuntimeError(f"Source changed externally during processing: {p['filename']}")
        report = build_report(timeline, settings, probes, diagnostics, preview, final)
        if social_final or social_preview:
            report += "\n## Social exports\n" + "\n".join(f"- {code}: {item['path']}" for code, item in (social_final or social_preview).items()) + "\n"
        (job / "report.md").write_text(report, encoding="utf-8")
        event("complete", 1, f"Finished {'4K reel and preview' if final else 'preview'}: {job}")
    except Cancelled as e:
        # The operation has ended and still owns the run lock. Clear its stale
        # command only here, never while a newer command could control a worker.
        control.unlink(missing_ok=True)
        event("cancelled", 0, str(e))
    except Exception as e:
        event("error", 0, str(e))
        raise
    finally:
        lock.close()

def build_report(timeline, settings, probes, diagnostics, preview, final):
    lines = ["# FPV Sesh run report", "", f"{len(timeline['shots'])} distinct shots • {timeline['duration']:.3f} seconds • {timeline['fps']} fps • {timeline.get('music_status', 'No music selected')}", "",
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
    lines += ["Full decode, frame count, rational frame rate, output dimensions, presentation timestamps and black-gap diagnostics saved separately. Original audio is decoded, trimmed, resampled and faded; music mixes use a lossless flight-sound intermediate before final AAC encoding. Music-mix details are in music-mix.json when music is selected. No invented drone sound.",
              "", f"Hardware: {diagnostics.get('gpu_name','see diagnostics')}; {diagnostics.get('vram_mib','unknown')} MiB VRAM; CPU and RAM details in logs/diagnostics.json.",
              "", "## Practical limits", "Selection uses motion, quality and image similarity heuristics. It cannot certify complete tricks or a semantic crash classification. Rotation is never itself treated as a crash. Gyro stabilization, synthetic slow motion and temporal restoration are disabled. Music timing extends only safe automatic exits toward clear detected beats; exact reviewed passages are preserved. Fully clipped sun detail cannot be recovered. Unknown color tags are explicitly reported. No claim of full-motion human review is made by the application.",
              "", *["- " + w for w in timeline["warnings"]], "", "Installed sources, licenses and benchmarks: application logs/backend-* and dependency records. This is a local editor and no media or metadata is uploaded."]
    return "\n".join(lines) + "\n"

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parser().parse_args()
    if args.command == "diagnose":
        data = hardware_diagnostics()
        try:
            from .enhance import backend_status
            data.update(backend_status())
        except (ImportError, AttributeError): pass
        save_json(ROOT / "logs" / "diagnostics.json", data)
        print(json.dumps(data, indent=2))
    elif args.command == "validate-ai":
        from .ai_validation import validate_ai
        result = validate_ai(args.input, args.start, args.seconds, lambda message: print(json.dumps({"stage": "ai-validation", "message": message}), flush=True))
        print(json.dumps(result, indent=2))
    elif args.command == "map-flight":
        from .video_review import map_flight
        map_flight(args.job, args.recognition)
    else: make(args)

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(json.dumps({"stage":"error", "progress":0, "message":str(exc)}), flush=True)
        sys.exit(1)
