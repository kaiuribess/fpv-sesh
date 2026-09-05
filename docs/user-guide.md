# FPV Sesh user guide

See [installation](../README.md#install-and-open) or [troubleshooting](troubleshooting.md). Press **F1** in the app for Help & setup.

## Your first edit without music

1. Open `launch.cmd`. In Session, choose **+ Add clips** or **Folder**. Files remain in their original locations; a fresh installation does not import personal recordings automatically.
2. Choose an edit style and short duration. Start with **Natural**, strength **0**, **Auto** quality and **Flight recognition → Off** for the smallest workflow.
3. Set **After preview → Stop at preview** and choose **Make my sesh**. Leave Music empty to use only source sound.
4. Watch the preview. In Moments, **Keep** or **Exclude** passages; use **Add exact range** for a missed interval. Choose **Regenerate edit** after changes.
5. Choose any additional shapes in Social, check their framing, and render the final. Use **Play 4K**, **Play selected export** or **Files** to open actual results.

## Session settings

| Setting | What changes |
| --- | --- |
| Energetic / Cinematic / Freestyle / Continuous | Varied highlights, more spacious passages, estimated action, or longer flight lines. Freestyle selection cannot certify complete tricks. |
| Duration | Auto aims for about 75 seconds and can grow for explicitly kept passages. Numeric choices are limits; short sessions are not padded. |
| Color look / strength | Fixed treatment for each passage. Natural at 0 preserves the source treatment. |
| Quality | Auto and Clean upscale use conventional scaling. AI detail requires optional setup and validation. |
| Order | Story flow arranges selected passages; Recording order keeps them chronological. |
| Codec | HEVC is the default master; H.264 supports more playback software. |
| After preview | Continue automatically or stop for review before final rendering. |
| Recovery seconds | Continued flight retained after estimated action, where available. |
| Flight recognition | Automatic uses an available compatible model; Off skips it; Thorough samples more densely. |

The master is 3840×2160. A 1440×1080 recording remains a 4:3 picture inside that canvas. Upscaling cannot restore native sensor detail, and Main10 output does not recover source bit depth. HDR/log conversion, non-square-pixel normalization, gyro stabilization and synthetic slow motion are not implemented in the main ingest path.

## Review moments

Moments shows source name, in/out times, ranking and selection evidence. **Rank** is an editing score, not a probability that a trick label is correct. Double-click a row, or focus it and press **Enter**, for the full explanation.

**Keep** requests a passage; **Exclude** prevents selection; **Clear overrides** clears those choices. **Add exact range** accepts seconds or `mm:ss`, checked against source duration. Include the approach, maneuver and exit when keeping a trick. Contradictory choices or impossible duration limits are reported rather than silently resolved.

Choose **Regenerate edit** after changing selection or settings. Regeneration uses the saved job's recordings. A changed input list is used for a new session when you choose Make my sesh. Final rendering preserves selected shot order and exact reviewed boundaries.

Optional **Teach this moment** labels are user confirmations, separate from predictions. They do not retrain the general video model. Local matching requires examples from different source identities and can abstain when examples disagree or are insufficient.

## Social exports

| Shape | Final dimensions |
| --- | --- |
| Landscape master | 3840×2160 |
| Vertical 9:16 | 1080×1920 |
| Square 1:1 | 1080×1080 |
| Portrait 4:5 | 1080×1350 |

**Full view with blurred background** preserves the entire picture over a blurred fill. **Full view with black bars** preserves it over black. **Crop to fill the frame** intentionally removes parts of the picture; adjust horizontal focus and inspect the preview around obstacles and maneuvers.

Social versions use original source intervals and the same edit order. Final social files use H.264/AAC. Playback lists only rendered files. Posting is manual; the app does not connect to social accounts. Platforms may recompress uploads.

## Music and source sound

Music is optional. Select a local decodable file such as MP3, WAV, M4A, AAC, FLAC or OGG. Track offset, level and fades are independent from original flight sound. Removing music clears a saved job's soundtrack choice too.

Full audio decode validation precedes rhythm analysis of up to 180 seconds after the offset. Silence or unclear rhythm produces no invented beat grid. Beat timing can extend a safe automatic exit by at most 0.8 seconds; it does not shorten exact reviewed intervals or sacrifice recovery to hit a beat. Some cuts remain off-beat.

Short music either fades or repeats the selected track tail with edge fades. This is not a professionally remixed seamless loop. The final soundtrack is peak-limited, encoded and checked for timing and sample peaks. Publish only music you have permission to use.

## Flight map

Motion and video observations occupy separate timeline bands. The table distinguishes **Estimate**, **Uncertain** and explicit user **Confirmed** evidence. Gray uncertainty is not confirmed acrobatics.

Filter by All motion, Possible tricks, Ordinary flight or Uncertain. Select an event and choose **Watch section** to play it with up to two seconds of surrounding footage on each side. Playback uses the local bundled player and creates no render. Double-click or press Enter on a selected row for the full method and evidence.

**Refresh understanding** analyzes a saved session without changing its cuts, music or finished videos. It verifies original file identities before attaching observations. Completed observations are cached; a partial scan is labeled accordingly and can be continued with another refresh. Restore missing originals first.

Automatic samples overlapping eight-second windows with up to eight frames per second. Thorough uses overlapping six-second windows with up to sixteen frames per second and a larger image budget. Neither examines every original frame. Motion tracking supplies separate image-rotation evidence, not the drone's physical attitude.

Recognition is experimental. A real trick can be missed, or ordinary flight can receive an incorrect label. There is no established FPV freestyle accuracy benchmark here. Review the source sequence before relying on roll, flip, split-S, powerloop, crash or recovery suggestions.

## Pause, cancel and saved jobs

Bottom controls remain available across pages. Pause requests a stop at the next supported stage, segment or frame boundary; the current operation can continue until then. Resume continues that paused process. Cancel retains completed outputs and usable cache entries.

The separate **Resume…** button selects an individual job folder under `output` and immediately continues it with saved settings. Closing during work asks whether to cancel safely and close. After a failure, inspect Activity before resuming; missing inputs and unsupported formats need their cause fixed first.

Choose **Files** for the current output folder. `preview.mp4` and `social-preview/` are smaller editing copies; `final_4k.mp4` and `social/` contain finished exports. `report.md` describes the run, `edit.csv` lists source intervals, and JSON files store settings, decisions, status and flight observations.

Back up the entire job folder and retain its originals at their saved locations. A finished MP4 is enough to watch, but rebuilding also requires the sources and saved decisions. Changed edits can retain earlier exports as `.previous.mp4`. The segment cache has a budget and is not a backup.

## Keyboard and Help

Use Tab / Shift+Tab between controls, arrow keys in lists and selection controls, and Enter or Space to activate the focused control. Scrolling pages bring focused controls into view. Clip cards support keyboard selection; Ctrl-click selects several with a mouse. Enter opens full evidence for a selected Moments or Flight map row.

F1 opens Help & setup with getting started, optional feature information, troubleshooting and local logs. Optional presence indicators are lightweight hints: they do not run inference or certify model integrity. The backend checks actual files before use.

## Optional models

Core editing works without model downloads. Install optional features only after an ordinary preview works. The optional stack uses a separate Python 3.12 environment; core Python 3.13 support does not imply optional-stack compatibility.

| Script | Adds |
| --- | --- |
| `setup-ai.ps1` | Torch/CUDA runtime and Real-ESRGAN image models. |
| `setup-vision.ps1` after AI setup | Places365 broad scene context; checkpoint about 46 MB. |
| `setup-video.ps1` after AI setup | Qwen3-VL general video interpretation; checkpoint about 4.3 GB plus packages. |

Read each setup message for current compatibility requirements. Initial Torch packages are several gigabytes. Scripts support `-CheckOnly`; individual checks may hash model files and perform a bounded inference. Video interpretation requires a compatible NVIDIA GPU and can run longer than the recording itself on an 8 GB GPU. That tested configuration is not a performance guarantee or a claimed minimum.

Real-ESRGAN uses native 2× restoration with a fixed 40% restored / 60% conventional blend. Validate and inspect a short sample before a long AI render: successful inference does not establish better texture or temporal quality. Auto remains conventional.

Places365 identifies broad surroundings, not trick identity. Qwen uses an existing general model; footage is not uploaded or scraped for new training. See [scene provenance](online-training.md), [video research](research-tricks.md) and [third-party notices](third-party.md).

## Command line

Open PowerShell in the application folder and use its own Python:

```powershell
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli make --folder input --recognition off --preview-only
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli make --job output/YOUR-JOB
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli make --job output/YOUR-JOB --no-music --preview-only
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli map-flight --job output/YOUR-JOB --recognition auto
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli validate-ai --input 'C:\Footage\flight.mp4' --start 20 --seconds 2
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli make --help
```

These are placeholder paths. `make --job` continues saved settings; `--no-music` removes a saved soundtrack. `map-flight` updates observations without rendering. See [CONTRIBUTING](../CONTRIBUTING.md) for tests.
