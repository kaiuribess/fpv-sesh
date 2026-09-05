# Local music analysis and soundtrack decisions

The application accepts local audio or the first audio stream of a video. It hashes every source byte, verifies a complete audio decode, and analyzes at most 180 seconds after the chosen offset. Cached analysis is tied to content, offset, and algorithm version. All processing stays local; NumPy and the existing FFmpeg installation are sufficient.

## Timing estimates

Onset strength measures increases in spectral energy. Tempo estimation and identifying beat positions are separate operations; a tempo alone does not locate cuts. Librosa's documentation describes onset measurement, tempo correlation, and selecting appropriately spaced peaks. Our small implementation uses those general signal-processing ideas without importing Librosa or reproducing its dynamic-programming tracker. [Librosa beat-tracking documentation](https://librosa.org/doc/latest/_modules/librosa/beat.html), [Librosa onset-strength definition](https://librosa.org/doc-playground/0.7.2/generated/librosa.onset.onset_strength.html).

We decode mono audio at 11,025 Hz, calculate positive log-spectral changes and energy transients with approximately 12 ms hops, estimate a 60–200 BPM pulse through autocorrelation, and return supported nearby onset peaks. Beat times are relative to the selected music offset. Silence, steady tones, and insufficient evidence produce no invented beat grid. Confidence is a heuristic reliability score, not a calibrated probability. Syncopation, half/double tempo, weak percussion, and tempo changes remain limitations. Video maneuver recovery should take precedence over snapping to music.

## Mixing choices

FFmpeg `afade` supplies timed fades. `amix` can normalize levels dynamically; we disable that behavior so ending music does not automatically amplify source sound. `alimiter` has optional makeup gain and lookahead latency compensation; we disable makeup and enable compensation. `loudnorm` supports measured two-pass normalization and true-peak targets, but it is deliberately not applied by default because the chosen workflow preserves fixed user gains. [Official FFmpeg audio-filter documentation](https://ffmpeg.org/ffmpeg-filters.html#amix).

Music gain accepts 0–1. Source sound arrives already leveled by the editor. The mixed signal uses a −2 dBFS sample limiter, then AAC at 48 kHz stereo and 256 kb/s. A complete decoded-output check verifies duration and sample peaks. Unusual AAC overshoot triggers one measured, fixed attenuation and a fresh encode from the inputs. This is sample-peak control, not a promise of a platform-specific LUFS target or true-peak mastering.

Short music defaults to a fade at its available end followed by source sound or silence. Optional looping repeats only the tail after the offset, with 20 ms edge fades and a warning that the joins may be audible. It does not claim a seamless musical rearrangement.

`analyze_music(...)` returns path, SHA256, duration, offset, beat estimates, BPM, confidence, and warnings. `mix_music(...)` writes `job/source-audio.m4a` only after a separate temporary file passes checks; `job/music-mix.json` records gains, short-track policy, warnings, and verification. The input may be the previous canonical source soundtrack without being overwritten while it is read. Cancel leaves completed output intact and removes only this call's temporary files.
