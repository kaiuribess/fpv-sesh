"""Local soundtrack analysis and mixing; no network, model, or paid service."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

import numpy as np

from .media import locate_tools, sha256_file

ANALYSIS_VERSION = 1
ANALYSIS_RATE = 11025
ANALYSIS_SECONDS = 180.0


def _finite(value, name, minimum=0):
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return value


def _execute(command, checkpoint, *, timeout=600, log=None):
    """Poll cooperative controls while native audio work runs without pipe stalls."""
    checkpoint()
    args = [str(value) for value in command]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(args, stdout=stdout, stderr=stderr, shell=False,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        started = time.monotonic()
        try:
            while process.poll() is None:
                checkpoint()
                if time.monotonic() - started > timeout:
                    raise TimeoutError("Audio processing exceeded its bounded runtime")
                time.sleep(.05)
            checkpoint()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        stdout.seek(0)
        stderr.seek(0)
        result = subprocess.CompletedProcess(args, process.returncode,
                                             stdout.read().decode("utf-8", "replace"),
                                             stderr.read().decode("utf-8", "replace"))
    if log is not None:
        with Path(log).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(args, ensure_ascii=False) + "\n" + result.stdout + result.stderr + "\n")
    if result.returncode:
        raise RuntimeError("Audio processing failed: " + (result.stderr or result.stdout)[-3000:])
    return result


def _audio_probe(path, checkpoint):
    _, ffprobe = locate_tools()
    result = _execute([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                      checkpoint, timeout=120)
    raw = json.loads(result.stdout)
    audio = next((stream for stream in raw.get("streams", []) if stream.get("codec_type") == "audio"), None)
    if audio is None:
        raise ValueError("The selected music file has no audio stream")
    return audio, raw.get("format", {})


def _decoded_duration(progress):
    values = re.findall(r"^out_time_us=(\d+)$", progress, re.MULTILINE)
    return max((int(value) / 1_000_000 for value in values), default=0)


def _waveform_features(samples, checkpoint):
    """Positive log-spectral changes and amplitude transients, at ~12 ms hops."""
    size, hop = 512, 128
    padded = np.pad(samples, (size // 2, size // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop]
    window = np.hanning(size).astype(np.float32)
    flux = np.zeros(len(frames), np.float32)
    energy = np.zeros(len(frames), np.float32)
    previous = np.zeros(size // 2 + 1, np.float32)
    for begin in range(0, len(frames), 1024):
        checkpoint()
        block = frames[begin:begin + 1024]
        energy[begin:begin + len(block)] = np.sqrt(np.mean(block * block, axis=1))
        spectrum = np.log1p(np.abs(np.fft.rfft(block * window, axis=1)) * 10)
        differences = np.diff(np.vstack((previous, spectrum)), axis=0)
        flux[begin:begin + len(block)] = np.maximum(differences, 0).mean(axis=1)
        previous = spectrum[-1]
    # Do not turn the file's artificial beginning/end into musical beats.
    flux[:3] = flux[-3:] = 0
    rising_energy = np.maximum(np.diff(energy, prepend=energy[0]), 0)
    if rising_energy.max(initial=0) > 0:
        flux += rising_energy / rising_energy.max() * flux.max(initial=0) * .35
    baseline_width = min(len(flux), max(1, round(.4 * ANALYSIS_RATE / hop)))
    baseline = np.convolve(flux, np.ones(baseline_width) / baseline_width, mode="same")
    envelope = np.maximum(flux - baseline * .6, 0)
    envelope[:3] = envelope[-3:] = 0
    return envelope, energy, hop / ANALYSIS_RATE


def _estimate_beats(samples, checkpoint):
    seconds = len(samples) / ANALYSIS_RATE
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Music decode contains no usable finite samples")
    if float(np.max(np.abs(samples))) < 1e-5:
        return [], None, 0.0, "The selected music window is silent; no beat timing was inferred"
    envelope, energy, step = _waveform_features(samples, checkpoint)
    interior = energy[3:-3] if len(energy) > 8 else energy
    stationary = float(np.std(interior)) / max(float(np.mean(interior)), 1e-8) < .025
    if seconds < 3 or stationary or float(envelope.max(initial=0)) < .002:
        return [], None, 0.0, "The selected music window has insufficient rhythmic evidence for beat timing"
    threshold = max(float(np.percentile(envelope, 85)) * .6, float(envelope.max()) * .08, .002)
    candidates = np.flatnonzero((envelope[1:-1] >= envelope[:-2]) &
                               (envelope[1:-1] > envelope[2:]) &
                               (envelope[1:-1] >= threshold)) + 1
    # Keep the strongest transient within each 120 ms neighborhood.
    accepted = []
    for index in candidates[np.argsort(envelope[candidates])[::-1]]:
        if all(abs(int(index) - existing) * step >= .12 for existing in accepted):
            accepted.append(int(index))
    peaks = np.array(sorted(accepted), dtype=int)
    if len(peaks) < 4:
        return [], None, 0.0, "Too few distinct onsets were found for reliable beat timing"
    checkpoint()
    fft_size = 1 << (2 * len(envelope) - 1).bit_length()
    spectrum = np.fft.rfft(envelope, n=fft_size)
    correlation = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_size)[:len(envelope)]
    low, high = max(2, round(60 / 200 / step)), min(len(envelope) - 2, round(60 / 60 / step))
    if high <= low or correlation[0] <= 0:
        return [], None, 0.0, "The music window is too short to estimate tempo"
    lags = np.arange(low, high + 1)
    prior = np.exp(-.5 * (np.log2((60 / (lags * step)) / 120) / .8) ** 2)
    lag = int(lags[np.argmax(correlation[lags] * prior)])
    left, center, right = correlation[lag - 1:lag + 2]
    denominator = left - 2 * center + right
    refinement = float(np.clip(.5 * (left - right) / denominator, -.5, .5)) if denominator else 0
    period = (lag + refinement) * step
    periodicity = float(np.clip(correlation[lag] / correlation[0], 0, 1))
    peak_times = peaks * step
    # Choose a phase supported by actual onsets, then follow local peaks near
    # that pulse. Missing pulses are not invented or returned as detected beats.
    anchor_options = peak_times[:min(len(peaks), 64)]
    def phase_score(anchor):
        phase = np.abs((peak_times - anchor + period / 2) % period - period / 2)
        return float(np.sum(envelope[peaks] * np.exp(-(phase / (period * .13)) ** 2)))
    anchor = float(max(anchor_options, key=phase_score))
    predicted = anchor % period
    beats, considered = [], 0
    while predicted < seconds:
        considered += 1
        distances = np.abs(peak_times - predicted)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= min(.09, period * .18):
            beat = float(peak_times[nearest])
            if not beats or beat > beats[-1] + period * .5:
                beats.append(beat)
                predicted = beat
        predicted += period
    confidence = float(np.clip((.6 * periodicity + .4 * len(beats) / max(considered, 1)) *
                               min(1, seconds / 8) * min(1, len(beats) / 8), 0, 1))
    if confidence < .2 or len(beats) < 4:
        return [], None, round(confidence, 3), "Rhythm is weak or irregular; automatic beat alignment is unavailable"
    warning = "Beat times and BPM are waveform estimates; syncopation, tempo changes, and half/double tempo can be misread"
    return [round(beat, 6) for beat in beats], round(60 / period, 2), round(confidence, 3), warning


def _cached_analysis(path, identity, offset):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        duration = float(result["duration"])
        beats = result["beats"]
        if (result.get("version") != ANALYSIS_VERSION or result.get("sha256") != identity
                or result.get("offset") != offset or not result.get("full_decode_validated")
                or not math.isfinite(duration) or not 0 <= offset < duration
                or not 0 <= float(result["confidence"]) <= 1
                or any(not math.isfinite(float(t)) or not 0 <= t < min(ANALYSIS_SECONDS, duration - offset) for t in beats)
                or any(a >= b for a, b in zip(beats, beats[1:]))):
            return None
        return result
    except (OSError, ValueError, TypeError, KeyError):
        return None


def analyze_music(path, cache, offset=0.0, checkpoint=lambda: None):
    """Fully validate audio, then estimate onsets in at most 180 s after offset.

    ``beats`` use soundtrack seconds: zero is the chosen source offset. A full
    content hash plus analysis version/offset identifies reusable cached results.
    """
    offset = _finite(offset, "Music offset")
    checkpoint()
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Music input must be a file")
    identity = sha256_file(source)
    checkpoint()
    directory = Path(cache) / "music"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{ANALYSIS_VERSION}:{identity}:{offset:.12g}".encode()).hexdigest()[:32]
    cache_path = directory / f"{key}.json"
    cached = _cached_analysis(cache_path, identity, offset)
    if cached is not None:
        return {**cached, "path": str(source), "cache_hit": True}
    audio, container = _audio_probe(source, checkpoint)
    ffmpeg, _ = locate_tools()
    validation = _execute([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror",
                           "-err_detect", "explode", "-i", source, "-map", "0:a:0", "-vn", "-sn",
                           "-af", "asetpts=N/SR/TB", "-progress", "pipe:1", "-nostats", "-f", "null", "-"],
                          checkpoint, timeout=1800)
    duration = _decoded_duration(validation.stdout)
    if duration <= 0:
        raise ValueError("The selected music file has no decodable audio duration")
    if offset >= duration:
        raise ValueError("Music offset must be before the end of the audio track")
    window_seconds = min(ANALYSIS_SECONDS, duration - offset)
    pcm = directory / f"analysis-{uuid.uuid4().hex}.f32"
    try:
        _execute([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror",
                  "-ss", str(offset), "-i", source, "-map", "0:a:0", "-vn", "-sn", "-t", str(window_seconds),
                  "-ac", "1", "-ar", str(ANALYSIS_RATE), "-c:a", "pcm_f32le", "-f", "f32le", pcm], checkpoint)
        samples = np.fromfile(pcm, dtype="<f4")
        beats, bpm, confidence, warning = _estimate_beats(samples, checkpoint)
    finally:
        pcm.unlink(missing_ok=True)
    warnings = [warning]
    if duration - offset > ANALYSIS_SECONDS:
        warnings.append("Beat analysis covers only the first 180 seconds after the music offset")
    result = {"version": ANALYSIS_VERSION, "path": str(source), "sha256": identity,
              "duration": round(duration, 6), "offset": offset, "beats": beats, "bpm": bpm,
              "confidence": confidence, "warnings": warnings, "analysis_seconds": round(window_seconds, 6),
              "beat_method": "positive log-spectral flux, energy onsets, tempo autocorrelation and local peak alignment",
              "full_decode_validated": True, "audio_codec": audio.get("codec_name", "unknown"),
              "sample_rate": audio.get("sample_rate"), "cache_hit": False}
    checkpoint()
    partial = cache_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        partial.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        partial.replace(cache_path)
    finally:
        partial.unlink(missing_ok=True)
    return result


def _verify_mix(path, ffmpeg, checkpoint, duration, log):
    result = _execute([ffmpeg, "-hide_banner", "-loglevel", "info", "-nostdin", "-xerror", "-i", path,
                       "-map", "0:a:0", "-af", "asetpts=N/SR/TB,astats=metadata=0:reset=0",
                       "-progress", "pipe:1", "-nostats", "-f", "null", "-"], checkpoint, log=log)
    actual = _decoded_duration(result.stdout)
    if abs(actual - duration) > .055:
        raise RuntimeError(f"Mixed soundtrack duration mismatch: {actual:.6f}s versus {duration:.6f}s")
    peaks = re.findall(r"Peak level dB:\s*([+-]?(?:\d+(?:\.\d+)?|inf))", result.stderr)
    peak = max((float(value) for value in peaks), default=None)
    if peak is None or math.isnan(peak):
        raise RuntimeError("Could not verify the decoded soundtrack sample peak")
    return {"decoded_duration": actual, "sample_peak_dbfs": peak if math.isfinite(peak) else None,
            "silent": peak == -math.inf, "full_decode_validated": True}


def mix_music(source_audio, music_info, job, duration, music_level, fade_seconds,
              short_policy="fade", checkpoint=lambda: None):
    """Mix fixed-gain music and pre-leveled source audio into the job soundtrack.

    Short music is faded then padded by default. ``loop`` repeats the selected
    tail with 20 ms edge fades; it is not described as a seamless musical edit.
    Source audio may be the canonical destination: replacement happens only
    after a separate temporary output is decoded and checked. Details and
    short-track warnings are saved in ``job/music-mix.json``.
    """
    duration = _finite(duration, "Soundtrack duration", .001)
    music_level = _finite(music_level, "Music level")
    fade_seconds = _finite(fade_seconds, "Music fade duration")
    if music_level > 1:
        raise ValueError("Music level must be between 0 and 1")
    if short_policy not in ("fade", "loop"):
        raise ValueError("Short music policy must be fade or loop")
    offset = _finite(music_info["offset"], "Music offset")
    track_duration = _finite(music_info["duration"], "Music duration", .001)
    if offset >= track_duration:
        raise ValueError("Music offset must be before the end of the audio track")
    checkpoint()
    music = Path(music_info["path"]).resolve(strict=True)
    if sha256_file(music) != music_info["sha256"]:
        raise ValueError("Music file changed after analysis; analyze it again before rendering")
    job = Path(job).resolve()
    target = job / "source-audio.m4a"
    if music == target:
        raise ValueError("The soundtrack output must never overwrite the selected music file")
    original = Path(source_audio).resolve(strict=True) if source_audio is not None else None
    job.mkdir(parents=True, exist_ok=True)
    ffmpeg, _ = locate_tools()
    token = uuid.uuid4().hex
    temporary = job / f"music-{token}.partial.m4a"
    loop_file = job / f"music-{token}.loop.wav"
    report_temp = job / f"music-{token}.report.tmp"
    log = job / "music-render.log"
    available = track_duration - offset
    short = available < duration - .025
    looping = short and short_policy == "loop"
    played = duration if looping else min(duration, available)
    fade = min(fade_seconds, played / 2)
    warnings = []
    if short:
        warnings.append("Music is shorter than the edit after the offset; " +
                        ("the selected tail repeats with short edge fades, so joins may be audible" if looping
                         else "the music fades at its end and the remaining edit has source sound or silence"))
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    try:
        if looping:
            edge = min(.02, available / 4)
            loop_filter = (f"aresample=48000,atrim=duration={available:.9f},asetpts=N/SR/TB,"
                           f"afade=t=in:d={edge:.9f},afade=t=out:st={available-edge:.9f}:d={edge:.9f}")
            _execute([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", str(offset),
                      "-i", music, "-map", "0:a:0", "-af", loop_filter, "-ac", "2", "-c:a", "pcm_f32le", loop_file],
                     checkpoint, log=log)
            command += ["-stream_loop", "-1", "-i", loop_file]
        else:
            command += ["-ss", str(offset), "-i", music]
        if original is not None:
            command += ["-i", original]
        music_filters = ["aresample=48000", f"atrim=duration={played:.9f}", "asetpts=N/SR/TB",
                         "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo", f"volume={music_level:.9f}"]
        if fade > 0:
            music_filters += [f"afade=t=in:d={fade:.9f}", f"afade=t=out:st={played-fade:.9f}:d={fade:.9f}"]
        music_filters += ["apad", f"atrim=duration={duration:.9f}"]
        graph = ["[0:a:0]" + ",".join(music_filters) + "[music]"]
        if original is not None:
            graph += [f"[1:a:0]aresample=48000,asetpts=N/SR/TB,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,apad,atrim=duration={duration:.9f}[source]",
                      "[music][source]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[mix]"]
            label = "[mix]"
        else:
            label = "[music]"
        # Fixed gain, no automatic makeup or loudness pumping. The conservative
        # sample ceiling leaves headroom for AAC reconstruction overshoots.
        final_filter = label + f"alimiter=limit=0.794328:level=false:latency=true,atrim=duration={duration:.9f},asetpts=N/SR/TB"
        graph.append(final_filter + "[out]")
        encode = command + ["-filter_complex", ";".join(graph), "-map", "[out]", "-vn", "-sn", "-ar", "48000", "-ac", "2",
                            "-c:a", "aac", "-b:a", "256k", "-t", f"{duration:.9f}", "-movflags", "+faststart", temporary]
        _execute(encode, checkpoint, log=log)
        verification = _verify_mix(temporary, ffmpeg, checkpoint, duration, log)
        peak = verification["sample_peak_dbfs"]
        if peak is not None and peak > -.1:
            # Re-encode from original inputs, not from already lossy AAC, using
            # a measured fixed attenuation if this material overshot the ceiling.
            attenuation = 10 ** ((-1.5 - peak) / 20)
            graph[-1] = final_filter + f",volume={attenuation:.9f}[out]"
            encode[encode.index("-filter_complex") + 1] = ";".join(graph)
            _execute(encode, checkpoint, log=log)
            verification = _verify_mix(temporary, ffmpeg, checkpoint, duration, log)
            warnings.append("A fixed final attenuation was applied after checking AAC sample-peak overshoot")
        peak = verification["sample_peak_dbfs"]
        if peak is not None and peak > -.1:
            raise RuntimeError("The encoded soundtrack still exceeds the checked sample-peak ceiling")
        report = {"music_path": str(music), "music_sha256": music_info["sha256"], "offset": offset,
                  "duration": duration, "music_available_seconds": available, "music_level": music_level,
                  "fade_seconds_requested": fade_seconds, "fade_seconds_applied": fade,
                  "short_music": short, "short_policy": short_policy, "looped": looping,
                  "source_audio_included": original is not None,
                  "gain_policy": "Fixed user music gain; pre-leveled source audio; no loudness normalization or automatic makeup",
                  "limiter_sample_ceiling_dbfs": -2.0, "codec": "aac", "bitrate": 256000,
                  "warnings": warnings, "verification": verification}
        report_temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        checkpoint()
        temporary.replace(target)
        report_temp.replace(job / "music-mix.json")
        return target
    finally:
        for owned in (temporary, loop_file, report_temp):
            owned.unlink(missing_ok=True)
