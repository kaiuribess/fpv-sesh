# FPV Sesh

A local Windows drone-highlight editor. Choose recordings, review selected moments, and export a lightweight preview plus a 3840×2160 session edit. Original footage stays untouched. Normal use requires no cloud account, API key, subscription, or Codex installation.

Music editing is disabled in this version. The editor can retain source sound at your chosen level; silent recordings remain silent.

## Install and launch

Install **64-bit Python 3.12 with Tkinter** from [python.org](https://www.python.org/downloads/windows/). Open PowerShell in this repository folder and run:

```powershell
& '.\setup.ps1'
& '.\launch.cmd'
```

If Python is not on your path, supply its location:

```powershell
& '.\setup.ps1' -PythonPath 'C:\path\to\python.exe'
```

Setup creates the isolated `.venv` environment, installs pinned packages from PyPI, and downloads the recorded FFmpeg 7.1.1 archive when needed. It verifies the recorded SHA256 before extraction. Windows `tar.exe` with 7z support is required. Setup does not install GPU drivers or change system settings or PowerShell policy. Use a PowerShell session permitted by your Windows policy.

After installation, double-click **launch.cmd** to open the application. To check an existing installation without installing anything:

```powershell
& '.\setup.ps1' -CheckOnly
```

GPU acceleration depends on the installed hardware and driver. Conventional rendering can fall back to CPU processing. Versions, download sources, licenses, and checksum limitations are recorded in `requirements-lock.txt`, `tools/dependencies.json`, and `tools/python-dependencies.json`.

## Make and review an edit

1. Add video files or choose a session folder. Folder selection includes compatible files directly inside that folder. No footage is included with the repository.
2. Choose **Sesh Hype**, **Cinematic Flow**, or **Freestyle Focus**, and Auto, 30, 60, 90, or 120 seconds. Choose a color look, strength, enhancement quality, source-sound level, and export codec.
3. Click **Make My Sesh**. The default workflow renders a 720p editing preview, then the final 4K file.
4. Use **Play edit preview (720p)** to review the edit and **Play final 4K** to assess the finished picture in your installed Windows video player. **Open output** opens the job folder.

Choose **Preview, then approve final** to review before final rendering. **Preview only** also stops after the preview. Once ready, click **Render final 4K**.

The **Review moments** table shows source intervals and estimated selection reasons. Double-click a row for details. Select rows with Ctrl or Shift, then use **Keep selected** or **Exclude selected**. **Regenerate edit** applies those choices and the current settings to that job's recordings. **Clear overrides** restores automatic selection. Pending changes must be regenerated before final rendering. Use **Make My Sesh** for a new job with a changed input list.

Selection uses motion, nearby-detail, exposure, and quality heuristics; it does not recognize or certify tricks. Persistent settling near the end of a recording can exclude arrival context automatically. Explicit **Keep** can restore a moment. Rotation alone is not treated as a crash.

Automatic endings try to preserve 2.5 seconds of continued footage after a detected rotation or strong vertical-motion burst. Linked bursts extend that interval. Incomplete endings remain available for explicit review. Reviewed source boundaries stay exact, including longer continuous passages. Auto aims for about 75 seconds and may grow to preserve kept passages; numeric duration choices remain limits. Weak or short sessions may produce shorter edits instead of filler.

## Picture quality and optional AI

Auto uses conventional libplacebo scaling on Vulkan when available, with Lanczos and CPU encoding fallbacks. Lanczos is also directly selectable. The output preserves the source's shape with padding rather than stretching or silently cropping it. For example, a 1440×1080 source becomes a 2880×2160 picture inside the 3840×2160 canvas. A 4K output does not imply native 4K camera detail.

HEVC is the default; H.264 offers broader playback compatibility. The render report identifies the actual backend, encoder, frame rate, and any fallback. GPU encoding is separate from AI enhancement.

An optional isolated CUDA Real-ESRGAN path uses native 2× restoration with a fixed 40% model / 60% conventional blend. It is slow and can smooth or alter fine texture, so Auto retains conventional scaling. To install or check its pinned environment and model downloads:

```powershell
& '.\setup-ai.ps1'
& '.\setup-ai.ps1' -CheckOnly
```

**AI setup alone does not enable AI on a fresh clone.** The application also requires a successful local video-validation record matching the model, code, runtime, and GPU. A supported command to create that record is not yet shipped in this repository. Machine-specific validation records are excluded from Git. Conventional editing remains available; an explicit AI request fails clearly when validation is unavailable.

The AI environment requires Windows 64-bit Python 3.12 and a compatible CUDA GPU. Dependencies and downloads are recorded in `requirements-ai-lock.txt`, `tools/ai-python-dependencies.json`, and `models/real-esrgan-cuda/manifest.json`. The older optional Video2X backend remains disabled after shutdown failures; `setup.ps1 -IncludeOptionalModels` installs its historical dependencies but does not enable it.

HDR, unsupported color interpretations, and non-square-pixel media are rejected. Gyro stabilization, music selection, beat matching, synthetic slow motion, titles, and advanced temporal restoration are not enabled.

## Pause, cancel, and resume

Pause and Cancel are cooperative. Conventional processing stops at supported stage or segment boundaries. CUDA restoration checks between frames, allowing an in-progress frame to finish. Click **Resume** to continue a paused process.

Cancel retains completed job files and cached segments; an incomplete AI segment is discarded. **Resume saved job…** reopens a job folder under `output` using its saved settings and checkpoints. Closing the interface during a render offers safe cancellation and waits for the renderer to stop.

Completed cache entries are checked against source identities, settings, and file hashes before reuse. Regenerating a changed edit moves older canonical videos to `.previous.mp4` so they are not presented as the new result.

## Outputs and storage

Each job under `output` contains its preview and final video when rendered, along with:

- `timeline.json` and `candidates.json`: selections, source intervals, timing, and review information.
- `settings.json`, `overrides.json`, and `status.json`: applied settings and recovery state.
- `report.md` and verification records: render choices, warnings, and technical checks.

The editor reads selected footage and writes application-owned job files. It does not upload videos or analysis data. A 40 GiB application-cache budget limits generated cache storage; automatic eviction is restricted to completed entries in `cache/segments`. Proxies count toward the budget but are not automatically evicted by that cleanup. Source recordings and job outputs are outside cache eviction.

**Repository scope:** Git contains application code, tests, setup scripts, dependency manifests, and license notices. Original footage, generated videos, job data, caches, logs, Python environments, downloaded tool binaries, and model weights stay local. Setup recreates required folders and downloads dependencies. A fresh clone contains no completed edits or machine-specific benchmark records.

## Command line and tests

Run commands from the repository folder after setup:

```powershell
# Preview recordings from a folder.
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli make --folder '.\input' --preview-only

# Make a preview and final from selected files.
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli make `
  --input 'C:\path\to\flight-one.mp4' `
  --input 'C:\path\to\flight-two.mp4' `
  --duration auto --style hype --look natural --strength 0 --quality auto

# Resume a saved job.
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli make --job '.\output\YOUR-JOB-FOLDER'

# See all current options.
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli make --help

# Run regression tests; media fixtures are generated locally.
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

To regenerate with saved review choices, add `--regenerate --overrides '.\output\YOUR-JOB-FOLDER\ui-overrides.json'`. The basic override format is:

```json
{"keep": ["candidate-id"], "exclude": ["another-candidate-id"]}
```

Tests cover media probing, rates and timestamps, source preservation, selection boundaries, cache recovery, CLI operation, and pause/cancel behavior. CUDA control tests use substitutes for the model and native processes; they do not validate GPU inference or visual quality.
