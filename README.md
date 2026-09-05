# FPV Sesh

A local Windows FPV editing studio. Import a flying session, review complete moments, add your own music, and create a 4K master plus social versions. Original recordings stay untouched on your computer.

## What is included

- A dark creator studio with real footage cards, a large source preview, dedicated editing pages, and persistent render controls.
- Four styles: energetic highlights, cinematic, freestyle tricks, and longer continuous flight.
- Story or recording order; Auto or 15–180 seconds; adjustable recovery after linked motion bursts.
- Keep/Exclude and exact source ranges, with reviewed boundaries preserved during final rendering.
- Local music files, independent sound levels, track offset, fades, optional looping, and conservative beat timing.
- A 3840×2160 master plus optional vertical 1080×1920, square 1080×1080, and portrait 1080×1350 versions.
- A temporal flight map with motion estimates, optional internet-pretrained scene understanding, and user-confirmed labels.
- Previews, verified outputs, pause/cancel, saved jobs, reusable segments, posters, and CSV edit decisions.

## Install and open

Install **64-bit Python 3.12 with Tkinter** from [python.org](https://www.python.org/downloads/windows/). Open PowerShell here, run `& '.\setup.ps1'`, then double-click **launch.cmd**.

If needed, use `setup.ps1 -PythonPath 'C:\path\to\python.exe'`. `setup.ps1 -CheckOnly` checks an existing installation. Setup creates an isolated environment, installs pinned packages, and verifies the downloaded FFmpeg archive. Windows tar with 7z support is required. GPU acceleration depends on hardware/driver support; conventional CPU fallbacks are available. No driver or PowerShell policy changes are made.

## Create and review

Add recordings, choose style/duration, optionally add music, and choose social shapes. Make a preview, inspect the moments and framing, then render the final files. The automatic workflow can create both stages in one job.

**Natural / 0%** is the default for faithful color. **Blur** preserves the full flight view over a blurred background; **Fit** uses bars; **Fill** deliberately crops with horizontal focus control. Social versions come directly from original source intervals.

Keep, Exclude, or Add exact range, then regenerate. Rendering a final preserves canonical shot order. Regeneration uses the job's recordings; changed input lists start a new session. Auto aims for about 75 seconds and may grow to retain explicitly kept passages in full. Numeric durations remain limits; short sessions are not padded with filler.

## Music

Choose MP3, WAV, M4A, AAC, FLAC, OGG, or another decodable audio source. Full decode validation precedes rhythm analysis of up to 180 seconds after your track offset. Silence or unclear rhythm produces no invented beat grid.

Beat timing can extend a safe automatic exit by at most 0.8 seconds. It never shortens a trick, changes an exact reviewed/user-kept interval, crosses another selected passage, or ignores recovery just to hit a beat. Some cuts intentionally remain off-beat. Looping repeats the selected track tail with edge fades; it is not a professionally remixed seamless loop.

Flight sound uses a lossless intermediate before music mixing. The final soundtrack is faded, peak-limited without makeup gain, encoded, decoded, and checked for timing and sample peaks. Details appear in music-mix.json. Removing music also removes a saved job's music choice.

## Online-pretrained flight context

Optional **Places365 ResNet18** uses weights pretrained on approximately 1.8 million internet scene images. It runs locally on proxy frames and supplies broad surroundings such as park/open grass, forest, sky, water, or built areas. Ambiguous scenes remain uncertain.

Run **setup-ai.ps1** for the shared optional Torch runtime, then **setup-vision.ps1** for the approximately 46 MB scene model. **setup-vision.ps1 -CheckOnly** verifies an existing installation and performs a small inference check. The initial Torch download is approximately 2.9 GB. Without these optional files, motion-based flight maps still work.

**Scene recognition is not a trained freestyle-trick detector.** It cannot prove a powerloop, double flip, crash, or geographic 3D route. Temporal motion adds evidence; user-confirmed labels remain distinct. Local matching needs multiple confirmed examples from different source identities and reports an independent-flight check. Replaced source files cannot inherit old confirmations. The app does not scrape or retrain on every online FPV video. See [online-model research](docs/online-training.md) for the actual evaluation, provenance, and attribution.

## Picture quality

The landscape master remains 4K. Conventional scaling, high-quality encoding, and restrained color are defaults. A 1440×1080 recording becomes a 2880×2160 picture inside the 4K canvas; output dimensions cannot create native camera detail. HEVC Main10 is the default master format; H.264 is also available. Social files use H.264/AAC.

Optional CUDA Real-ESRGAN uses native 2× restoration with a fixed 40% restored / 60% conventional blend. After setup-ai.ps1, run `python -m fpvsesh.cli validate-ai --input "C:\path\to\flight.mp4" --start 20 --seconds 2` using the project's .venv Python. It renders and verifies real frames and records the exact model, code, runtime, GPU, encoder, and probe identities. Inspect the sample before a long edit: successful inference does not certify natural texture or temporal quality.

HDR/log interpretation, non-square-pixel normalization, gyro stabilization, synthetic slow motion, and general temporal restoration are not implemented in the main ingest path. The old Video2X adapter remains disabled.

## Outputs and recovery

Each job under output contains preview.mp4, final_4k.mp4 when rendered, and requested files under social-preview/ and social/. Supporting files include timeline.json, candidates.json, flight-map.json, edit.csv, settings.json, overrides.json, status.json, exports.json, report.md, verification records, posters, and publish-notes.md.

Use final files for posting; previews are smaller editing copies. Platforms can recompress uploads. No social-account publishing occurs.

Pause/cancel takes effect at supported stage/segment boundaries; CUDA restoration checks between frames. Cancel retains completed cache entries. Resume a job folder to continue. Changed edits retire old canonical videos to .previous.mp4. Sources and job outputs are outside the 40 GiB segment-cache eviction policy.

## Command line and tests

Use the Python executable under .venv/Scripts from this folder:

- `python -m fpvsesh.cli make --folder input --music "C:\path\to\track.mp3" --social-formats vertical,square,portrait --framing blur --preview-only`
- `python -m fpvsesh.cli make --job output/YOUR-JOB` resumes saved settings and reviewed order.
- `python -m fpvsesh.cli make --job output/YOUR-JOB --no-music --preview-only` removes saved music.
- `python -m fpvsesh.cli make --help` lists all options.
- `python -m pytest -q` runs regression tests.

Tests generate media and check timing, music, framing, cache recovery, source preservation, UI/CLI settings, and model gating. They do not certify artistic quality or named-trick accuracy. Actual-camera and optional-model checks are recorded locally.

## Research and repository scope

Read the [FPV research](docs/research-fpv.md), [music notes](docs/research-music.md), [social guidance](docs/research-social.md), [online-model evaluation](docs/online-training.md), and [UI design brief](docs/ui-design.md).

Git contains code, tests, setup scripts, manifests, and license notices. Footage, music, renders, jobs, diagnostics, environments, downloaded tools, and weights stay local. Upstream models/libraries keep their separate licenses. Ordinary editing requires no account, API key, subscription, or cloud inference.
