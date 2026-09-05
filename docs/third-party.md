# Third-party components

FPV Sesh's own application and documentation use the [MIT license](../LICENSE). Third-party code, tools and model weights keep their separate terms. This source archive contains notices and manifests; setup downloads external binaries and checkpoints separately.

| Component | Role and attribution |
| --- | --- |
| FFmpeg / FFprobe / FFplay | Local media decode, encode, inspection and section playback. The selected full build is GPL-3.0-or-later; its exact version, builder, upstream source and hashes are recorded in `tools/dependencies.json`. See [FFmpeg licensing](https://ffmpeg.org/legal.html). |
| Python packages | Core and optional environments have separate exact locks and provenance records under `tools/`. Installed wheels retain their own notices; optional notice copies are under `tools/ai-license-notices` and `tools/video-license-notices`. |
| Real-ESRGAN | Optional restoration weights and adapted architecture; BSD-3-Clause. The selected RRDB architecture also includes adapted BasicSR code under Apache-2.0. Complete notices and changes are recorded under `models/real-esrgan-cuda`. |
| Places365 | Optional scene checkpoint from MIT CSAIL. Upstream identifies an attribution CC BY weight license without a version; this project does not invent one. Separate MIT/BSD code notices and author credit are in [MODEL-LICENSE.md](../models/places365/MODEL-LICENSE.md). |
| Qwen3-VL-2B-Instruct | Optional general video-language checkpoint from Qwen / Alibaba Cloud under Apache-2.0. Exact revision, publisher digest, license and model limitations are recorded under [models/qwen3-vl-2b](../models/qwen3-vl-2b/MODEL-NOTES.md). |
| Video2X | Historical, disabled experimental adapter under AGPL-3.0-or-later. It is not required for normal editing; the retained upstream notice does not mean it is active. |

Manifest hashes identify the recorded assets. Some historical model digests were calculated after official downloads because no independent publisher digest was available; their manifests disclose that limitation. A matching local hash is not a claim of publisher signature verification.

Before redistributing external binaries or weights, review the relevant upstream terms and corresponding-source requirements. The application's MIT license does not replace those terms. Music and input footage remain the responsibility of their owners and are not included in the source release.
