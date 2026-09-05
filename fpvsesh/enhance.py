"""Local selected-interval enhancement adapters. Models never receive unrelated shots."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / 'tools/ffmpeg-7.1.1/ffmpeg-7.1.1-full_build/bin/ffmpeg.exe'
FFPROBE = FFMPEG.with_name('ffprobe.exe')
VIDEO2X = ROOT / 'tools/video2x-6.4.0/video2x.exe'
MODEL_VERSION = 'video2x-6.4.0/realesrgan-plus-x4'
CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
BENCHMARK_SIGNATURE = ROOT / 'logs/backend-tested-signature.json'


def _detected_gpu() -> dict:
    result = {'uuid': None, 'name': 'Not detected', 'vram_mib': None, 'driver_version': None}
    try:
        detected = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
                                   '--format=csv,noheader,nounits'], capture_output=True,
                                  text=True, timeout=5, creationflags=CREATE_FLAGS)
        gpu_id, name, memory, driver = detected.stdout.strip().splitlines()[0].split(',')
        result = {'uuid': gpu_id.strip(), 'name': name.strip(),
                  'vram_mib': int(memory), 'driver_version': driver.strip()}
    except (ValueError, OSError, IndexError, subprocess.TimeoutExpired):
        pass
    return result


def _tool_hashes(*, ffmpeg=None, ffprobe=None) -> dict:
    """Compare actual binaries and model files with the saved local benchmark."""
    paths = [FFMPEG, FFPROBE, VIDEO2X,
             VIDEO2X.parent / 'models/realesrgan/realesrgan-plus-x4.bin',
             VIDEO2X.parent / 'models/realesrgan/realesrgan-plus-x4.param']
    paths += sorted(VIDEO2X.parent.glob('*.dll'))
    hashes = {}
    for path in paths:
        actual_path = path
        if path == FFMPEG and ffmpeg is not None:
            actual_path = Path(ffmpeg)
        elif path == FFPROBE and ffprobe is not None:
            actual_path = Path(ffprobe)
        try:
            with actual_path.open('rb') as handle:
                hashes[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.file_digest(handle, 'sha256').hexdigest()
        except OSError:
            hashes[str(path.relative_to(ROOT)).replace('\\', '/')] = None
    return hashes


def backend_status(*, ffmpeg=None, ffprobe=None) -> dict:
    """Gate recorded measurements by the exact GPU, driver, and tool hashes."""
    gpu = _detected_gpu()
    from .cuda_backend import status as cuda_status
    cuda = cuda_status(gpu, ffmpeg=ffmpeg, ffprobe=ffprobe)
    try:
        saved = json.loads(BENCHMARK_SIGNATURE.read_text(encoding='utf8'))
    except (OSError, ValueError):
        saved = {}
    matches_hardware = bool(saved.get('gpu')) and gpu == saved['gpu']
    matches_tools = bool(saved.get('tool_sha256')) and _tool_hashes(ffmpeg=ffmpeg, ffprobe=ffprobe) == saved['tool_sha256']
    matches_tested = matches_hardware and matches_tools
    measured = saved.get('measurements', {}) if matches_tested else {}
    warnings = ([
        'Real-ESRGAN GPU inference produced 60 frames, but Video2X crashed during teardown in two runs; normal AI rendering is disabled.',
        'Native 4x Real-ESRGAN measured 0.156 fps before 4K downsampling (roughly 6.4 hours per minute of 60 fps footage).',
    ] if matches_tested else [
        'The detected GPU, driver, or tool hashes differ from the saved benchmark (or no saved benchmark exists). Encoder, inference, and speed measurements are not verified for this configuration.',
        'AI remains unavailable. Run local diagnostics before treating any GPU backend as tested on this computer.',
    ])
    warnings.append('FlashVSR was not installed: its required official sparse-attention backend documents Linux; native Windows 8GB compatibility remains unverified.')
    if cuda["available"]:
        warnings = ["The old Video2X backend remains disabled. The current AI path uses independently tested native 2x CUDA restoration with a restrained source blend."]
    return {
        'gpu_name': gpu['name'],
        'vram_mib': gpu['vram_mib'],
        'driver_version': gpu['driver_version'],
        'matches_tested_hardware': matches_hardware,
        'matches_tested_tool_hashes': matches_tools,
        'benchmark_verified': matches_tested,
        'vulkan_api': measured.get('vulkan_api'),
        'encoder': measured.get('encoder', 'Not benchmarked for the detected configuration'),
        'upscale_backend': measured.get('upscale_backend', 'Not benchmarked for the detected configuration'),
        'ai_available': cuda['available'],
        'ai_inference_demonstrated': cuda['available'] or matches_tested,
        'ai_benchmark_fps': cuda['fps'] if cuda['available'] else measured.get('ai_benchmark_fps'),
        'ai_peak_total_gpu_memory_mib': cuda['peak_total_gpu_memory_mib'] if cuda['available'] else measured.get('ai_peak_total_gpu_memory_mib'),
        'ai_profile': cuda['profile'] if cuda['available'] else None,
        'version': 'ffmpeg-7.1.1/cuda-real-esrgan-v1' if cuda['available'] else 'ffmpeg-7.1.1/libplacebo; video2x-6.4.0-ai-disabled',
        'warnings': warnings,
    }


def _probe(path: Path, ffprobe: Path) -> dict:
    p = subprocess.run([str(ffprobe), '-v', 'error', '-select_streams', 'v:0',
                        '-show_streams', '-show_format', '-of', 'json', str(path)],
                       capture_output=True, text=True, creationflags=CREATE_FLAGS)
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])
    data = json.loads(p.stdout)
    if not data.get('streams'):
        raise ValueError('The selected file has no video stream.')
    return {**data['streams'][0], '_format': data.get('format', {})}


def _run(command: list, name: str, log, timeout: float = 3600) -> dict:
    """Keep native process windows hidden; poll GPU memory, not inference guesses."""
    log_path = ROOT / 'logs' / f'backend-{name}-{uuid.uuid4().hex[:8]}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    samples, start, last_report = [], time.monotonic(), 0.0
    with log_path.open('w', encoding='utf8') as handle:
        handle.write(json.dumps([str(x) for x in command]) + '\n')
        handle.flush()
        process = subprocess.Popen([str(x) for x in command], stdout=handle,
                                   stderr=subprocess.STDOUT, creationflags=CREATE_FLAGS)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    raise TimeoutError(f'{name} exceeded {timeout:.0f} seconds')
                try:
                    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                                          '--format=csv,noheader,nounits'],
                                         capture_output=True, text=True, timeout=3,
                                         creationflags=CREATE_FLAGS)
                    samples.append(int(gpu.stdout.strip().splitlines()[0]))
                except (ValueError, OSError, subprocess.TimeoutExpired, IndexError):
                    pass
                if elapsed - last_report >= 20:
                    log(f'{name}: processing selected interval ({elapsed:.0f}s elapsed)')
                    last_report = elapsed
                time.sleep(.3)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    result = {'elapsed_seconds': round(time.monotonic()-start, 3),
              'peak_total_gpu_memory_mib': max(samples, default=None),
              'memory_measurement': 'Total GPU usage, including desktop and other applications; sampled, not process-exclusive.',
              'log_path': str(log_path), 'returncode': process.returncode}
    if process.returncode:
        raise RuntimeError(f'{name} failed: ' + log_path.read_text(encoding='utf8')[-2400:])
    return result


def _fit_dimensions(source: dict, width: int, height: int) -> tuple[int, int]:
    sar = source.get('sample_aspect_ratio', '1:1')
    try:
        ratio = float(Fraction(sar.replace(':', '/')))
        if ratio <= 0:
            ratio = 1.0
    except (ValueError, ZeroDivisionError):
        ratio = 1.0
    aspect = source['width'] * ratio / source['height']
    # FFmpeg's default autorotation runs before our scaler. A quarter turn
    # swaps displayed axes (and reciprocates non-square-pixel display aspect).
    # Display-matrix metadata takes precedence over the legacy rotate tag.
    rotation = next((item['rotation'] for item in source.get('side_data_list', [])
                     if 'rotation' in item), source.get('tags', {}).get('rotate', 0))
    try:
        rotation = float(rotation) % 360
        if math.isclose(rotation, 90, abs_tol=.01) or math.isclose(rotation, 270, abs_tol=.01):
            aspect = 1 / aspect
    except (TypeError, ValueError):
        pass
    if width / height > aspect:
        return max(2, int(height * aspect) // 2 * 2), height
    return width, max(2, int(width / aspect) // 2 * 2)


def enhance_segment(input_path: Path, output_path: Path, settings: dict, log=print) -> dict:
    """Render one contiguous selection, video only, at a documented timeline rate.

    settings: start, duration, frames, fps (rational string), grade (FFmpeg filter),
    quality auto|lanczos|ai, codec hevc|h264, width/height, and optional tool paths.
    Auto means measured conventional libplacebo Vulkan. Explicit AI selects the
    separately validated native 2x CUDA model with a restrained source blend.
    The caller owns global cache identity and cancellation between intervals.
    """
    source_path, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if source_path == destination:
        raise ValueError('An output must never overwrite its source.')
    ffmpeg = Path(settings.get('ffmpeg', FFMPEG))
    ffprobe = Path(settings.get('ffprobe', ffmpeg.with_name('ffprobe.exe')))
    source = _probe(source_path, ffprobe)
    width, height = int(settings.get('width', 3840)), int(settings.get('height', 2160))
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError('Output dimensions must be positive even integers.')
    quality = settings.get('quality', 'auto')
    if quality == 'conventional':
        quality = 'auto'
    if quality not in ('auto', 'lanczos', 'ai'):
        raise ValueError(f'Unsupported enhancement quality: {quality}')
    start = float(settings.get('start', 0))
    duration = float(settings.get('duration', source.get('duration', source['_format'].get('duration', 0))))
    fps = str(settings.get('fps', source.get('avg_frame_rate', '60/1')))
    rate = Fraction(fps)
    if start < 0 or duration <= 0 or not math.isfinite(start + duration) or rate <= 0:
        raise ValueError('Invalid source interval or timeline frame rate.')
    frames = int(settings.get('frames', round(duration * float(rate))))
    if frames < 1:
        raise ValueError('An interval must contain at least one output frame.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < 256 * 1024**2:
        raise OSError('Less than 256 MiB of free output space remains.')
    job_id = uuid.uuid4().hex[:10]
    partial = destination.with_name(destination.stem + f'.working-{job_id}' + destination.suffix)
    temp_files, warnings, stages = [], [], []
    actual_source, actual_start, actual_quality = source_path, start, quality
    if quality == 'ai':
        if not backend_status(ffmpeg=ffmpeg, ffprobe=ffprobe)['ai_available']:
            raise RuntimeError('AI enhancement is not validated for the current environment; choose conventional scaling or run its local validation first.')
        from .cuda_backend import render as render_cuda
        fit = _fit_dimensions(source, width, height)
        configuration = {**settings, 'ffmpeg': str(ffmpeg), 'ffprobe': str(ffprobe), 'start': start, 'duration': duration,
                         'frames': frames, 'fps': fps, 'width': width, 'height': height,
                         'rate_conversion': bool(settings.get('vfr')) or Fraction(source['avg_frame_rate']) != rate}
        try:
            result = render_cuda(source_path, partial, configuration, source, fit, log)
            verified = _probe(partial, ffprobe)
            if (verified['width'], verified['height']) != (width, height) or int(verified.get('nb_frames', -1)) != frames or Fraction(verified['avg_frame_rate']) != rate:
                raise RuntimeError('AI output dimensions, frames or frame rate do not match the timeline')
            partial.replace(destination)
            result['output_path'] = str(destination)
            destination.with_suffix(destination.suffix + '.enhance.json').write_text(json.dumps(result, indent=2), encoding='utf8')
            return result
        finally:
            partial.unlink(missing_ok=True)
    try:
        fit_w, fit_h = _fit_dimensions(source, width, height)
        codec = settings.get('codec', 'hevc')
        if codec not in ('hevc', 'h264'):
            raise ValueError('The tested encoders are HEVC or H.264.')
        source_rate = source.get('avg_frame_rate', '0/0')
        source_nominal = source.get('r_frame_rate', source_rate)
        conversion = source_rate != fps or source_nominal != source_rate or bool(settings.get('vfr'))
        try:
            conversion = Fraction(source_rate) != rate or Fraction(source_nominal) != Fraction(source_rate) or bool(settings.get('vfr'))
        except (ValueError, ZeroDivisionError):
            conversion = True
        rate_filter = [f'fps={fps}:round=near'] if conversion else []
        grade = settings.get('grade', '')
        post = ['setsar=1']
        if grade:
            post.append(str(grade))
        post.append(f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black')
        post += ['setpts=PTS-STARTPTS'] + rate_filter
        output_pix = 'p010le' if codec == 'hevc' else 'yuv420p'
        backend = 'lanczos-cpu' if actual_quality == 'lanczos' else 'libplacebo-vulkan'
        encoder = 'hevc_nvenc' if codec == 'hevc' else 'h264_nvenc'
        trials = [(backend, encoder)]
        if backend != 'lanczos-cpu':
            trials.append(('lanczos-cpu', encoder))
        trials.append(('lanczos-cpu', 'libx265' if codec == 'hevc' else 'libx264'))
        for attempt, (backend, encoder) in enumerate(trials):
            gpu_scale = backend == 'libplacebo-vulkan'
            command = [ffmpeg, '-hide_banner', '-y']
            if gpu_scale:
                command += ['-init_hw_device', 'vulkan=vk:0', '-filter_hw_device', 'vk']
            command += ['-ss', str(actual_start), '-i', actual_source, '-map', '0:v:0', '-an']
            scaler = f'libplacebo=w={fit_w}:h={fit_h}:upscaler=ewa_lanczos:downscaler=mitchell:format=yuv420p10le,format=yuv420p10le' if gpu_scale else f'scale={fit_w}:{fit_h}:flags=lanczos,format=yuv420p10le'
            command += ['-vf', ','.join([scaler] + post + [f'format={output_pix}']), '-frames:v', str(frames)]
            if encoder.endswith('_nvenc'):
                command += ['-c:v', encoder, '-preset', 'p7', '-tune', 'hq', '-rc', 'vbr', '-cq', str(settings.get('cq', 16)), '-b:v', '0', '-spatial-aq', '1', '-temporal-aq', '1', '-rc-lookahead', '32', '-multipass', 'fullres']
                if codec == 'hevc':
                    command += ['-profile:v', 'main10', '-tag:v', 'hvc1', '-maxrate', '120M', '-tier', 'high']
            else:
                command += ['-c:v', encoder, '-preset', 'fast', '-crf', '18']
                if encoder == 'libx264':
                    command += ['-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709']
            # Set the encoder/timeline rate explicitly: FFV1 + libplacebo can
            # otherwise lose rate metadata and make FFmpeg default to 25fps.
            # This output option does not reinterpret source timestamps.
            command += ['-r', fps, '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv', '-fps_mode', 'cfr']
            if destination.suffix.lower() in ('.mp4', '.mov'):
                command += ['-movflags', '+faststart', '-video_track_timescale', str(rate.numerator)]
            command += [partial]
            try:
                log(f'Rendering selected interval with {backend} and {encoder}')
                stages.append(_run(command, backend + '-' + encoder, log, timeout=max(1800, frames*5)))
                break
            except (RuntimeError, TimeoutError) as error:
                if attempt + 1 == len(trials):
                    raise
                warnings.append(f'{backend}/{encoder} failed; trying {trials[attempt+1][0]}/{trials[attempt+1][1]}: {error}')
                log(warnings[-1])
        final_probe = _probe(partial, ffprobe)
        if (final_probe['width'], final_probe['height']) != (width, height):
            raise RuntimeError('Rendered segment dimensions do not match the requested output.')
        if Fraction(final_probe.get('avg_frame_rate', '0/1')) != rate:
            raise RuntimeError('Rendered segment frame rate does not match the requested timeline.')
        actual_frames = final_probe.get('nb_frames')
        if actual_frames and int(actual_frames) != frames:
            raise RuntimeError(f'Rendered segment has {actual_frames} frames; expected {frames}.')
        partial.replace(destination)
        result = {'output_path': str(destination), 'backend': backend, 'encoder': encoder,
                  'ai_inference': quality == 'ai' and actual_quality == 'ai',
                  'model': MODEL_VERSION if actual_quality == 'ai' else None,
                  'native_model_scale': 4 if actual_quality == 'ai' else None,
                  'scaling_note': 'Native 4x model inference, then conventional downsample to contain within 4K.' if actual_quality == 'ai' else 'Conventional scaling; no AI inference.',
                  'width': width, 'height': height, 'content_width': fit_w, 'content_height': fit_h,
                  'fps': fps, 'frames': frames, 'frame_rate_conversion': conversion,
                  'start': start, 'duration': frames / float(rate), 'source_path': str(source_path),
                  'warnings': warnings, 'stages': stages,
                  'elapsed_seconds': round(sum(s['elapsed_seconds'] for s in stages), 3),
                  'peak_total_gpu_memory_mib': max((s['peak_total_gpu_memory_mib'] or 0 for s in stages), default=0)}
        destination.with_suffix(destination.suffix + '.enhance.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        return result
    finally:
        for temporary in [partial] + temp_files:
            # Only exact, unique application-owned files created by this call.
            if temporary.exists():
                temporary.unlink()
