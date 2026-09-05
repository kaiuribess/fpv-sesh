# Troubleshooting

Press **F1 → Help & setup** in the app. Activity contains progress, warnings and the reason a job stopped. Double-click **doctor.cmd** for local readiness information. Fully extract the source ZIP first and keep its files together in a writable local folder.

## Setup or launch stops

| Symptom | Next step |
| --- | --- |
| Python missing or wrong version | Install supported 64-bit Python with Tkinter. Core supports 3.12 and 3.13; the recommended current Windows installer is [3.13.15](https://www.python.org/downloads/release/python-31315/). Run install.cmd again. |
| Several Python installations | Run `setup.ps1 -PythonPath 'C:\Python\python.exe'`, substituting the supported executable's actual path. |
| Tkinter missing | Use the full Windows installer with Tcl/Tk enabled; the embeddable ZIP is not the desktop runtime. |
| Folder is not writable | Extract a fresh copy to a folder you own outside Program Files. |
| Download fails | Keep the setup message and retry when the official host is reachable. Follow your administrator's policy on a managed network. |
| Checksum mismatch | Stop using that download and retry from a fresh source release. Do not edit expected hashes or disable verification. |
| Archive extraction fails | Check the exact setup message and required extraction tool; a partial archive cannot be repaired by renaming it. |
| Existing environment incomplete | Preserve jobs and recordings. Verify a fresh application copy before retiring the incomplete copy. |
| launch.cmd stops | Read its message, rerun install.cmd, and inspect local startup logs if the problem remains. |

For a quoted PowerShell script path, use `& '.\setup.ps1'`. A managed device may prohibit scripts; do not change organization security policy to bypass its restriction. Setup does not install GPU drivers or change system security settings.

## GPU missing or render slow

GPU encoding, scaling and inference are separate capabilities. A detected GPU does not guarantee every codec or model works. Activity and the run report identify the actual path. CPU fallbacks can be much slower.

Try Auto or Clean upscale and a short preview. Optional AI detail costs substantially more time. If a driver is incompatible, consult the hardware vendor or administrator before changing it. No driver change is performed by this application.

## Recognition unavailable, partial or uncertain

Core editing and motion estimates work without optional models. Choose Flight recognition → Off to skip video interpretation. Help's file-presence hint is not an integrity or inference test; the backend explains missing files, incompatible runtimes and checksum failures.

A partial scan keeps completed observations. Restore missing originals, resolve the reported cause, and choose Flight map → Refresh understanding. This updates observations without changing your finished edit. If GPU memory is exhausted, close other heavy GPU applications and try Automatic before Thorough.

Uncertain means the evidence does not support a stronger label; it is not an installation failure. Watch the entire section. General video interpretation can miss tricks or suggest them during ordinary flight. More sampling does not guarantee the right answer.

## A trick is missing or cut too early

Use Moments → Add exact range to keep the approach, maneuver and exit, then regenerate. Increase recovery time where useful for automatic passages. If a numeric duration cannot fit your kept passages, choose a longer duration or Auto. Resolve contradictory keep/exclude choices when reported. Beat timing does not override exact reviewed boundaries.

## Originals moved or replaced

Saved jobs identify both file path and original bytes. A different recording at the same path cannot inherit old analysis or confirmations. Restore the original at its saved location to continue that job; use a new session for different footage. Do not change saved hashes to force reuse.

Resume… expects an individual job folder with status or timeline files and immediately continues it. A finished MP4 plays independently, but rebuilding requires the original recordings and job decisions.

## Pause or cancel takes time

The request takes effect at the next supported stage, segment or frame boundary. Wait for the paused or cancelled message. After an unexpected exit, completed cache entries remain available; inspect Activity before resuming. Avoid running multiple workers against one application folder.

## No final video or playback disabled

Stop at preview and Wait for final approval intentionally stop before final rendering. Review the preview, regenerate pending changes and choose Render final 4K. Playback only offers existing files, and pending edit settings can disable final playback until rendered.

Open Files and read `report.md` if processing stopped. An older preview can remain after failure; its existence does not establish that the latest final succeeded. If Windows cannot decode a valid HEVC file, use a compatible installed player or render H.264. Watch section uses the bundled player; missing FFplay/FFmpeg tools require setup repair.

## Picture or sound is unexpected

Start with Natural at 0 strength and conventional scaling. Full view with blurred background or black bars preserves the picture; Crop to fill intentionally removes edges. Review social framing around obstacles. Upscaling cannot restore native sensor detail, and AI can smooth or invent texture.

The main ingest path does not implement HDR/log conversion, non-square-pixel normalization or gyro stabilization. Unsupported interpretations are reported rather than silently treated as ordinary SDR.

For music, check file, level, offset and end behavior. The offset must be inside the track. Silence or unclear rhythm produces no beat grid. Source audio and music levels are independent; a silent source has no sound to turn up. Read the music warning and `music-mix.json` for decode or mix failures.

## Report a problem

Use the [bug report form](https://github.com/kaiuribess/fpv-sesh/issues/new/choose). Include release/commit, Windows and Python versions, GPU/driver if relevant, chosen settings and the shortest reproduction steps. Describe expected and actual behavior.

Review logs and screenshots before sharing. Remove home-directory paths, personal names, tokens, GPS coordinates and recognizable private property. Do not upload complete jobs, caches, original recordings or music by default. Prefer synthetic media. Share essential real footage only with permission and through an agreed suitable channel.

For a suspected vulnerability, follow [SECURITY.md](../SECURITY.md) instead of posting exploit details publicly.
