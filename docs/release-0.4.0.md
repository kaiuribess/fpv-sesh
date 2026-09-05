# FPV Sesh 0.4.0

A Windows source release for turning FPV recordings into reviewed highlights, a 4K master, and vertical, square and portrait social versions. Music is optional and stays local.

## Start here

Download **FPV-Sesh-0.4.0-Windows-source.zip**, extract the whole folder to a writable local location, and install 64-bit Python 3.13.15 with Tcl/Tk from [python.org](https://www.python.org/downloads/release/python-31315/). Double-click **install.cmd**, then **launch.cmd**. Press **F1** for the guide. **doctor.cmd** checks local readiness and saves a report designed for sharing without personal paths.

This is a source distribution, not a signed standalone executable. Setup downloads hash-verified dependencies and FFmpeg from their recorded publishers. Models are separate, optional downloads through **install-models.cmd**. Core editing needs no account, API key, model download or subscription.

## Release improvements

- Clean first-run guidance, Help, keyboard navigation, visible startup errors and honest optional-feature status.
- Verified, repairable installation; exact package versions and wheel hashes; maintained FFmpeg 9.0.1; a matching verified player and probe.
- Failed source validation preserves a saved job's settings and music decisions. Incomplete, changed or truncated analysis caches are rebuilt.
- Shared job locking and interruption controls; reviewed source ranges survive resume and music changes.
- CPU encoding fallback for incompatible GPU encoders, including the optional CUDA restoration path. Actual encoder and quality settings are recorded.
- Updated optional AI packages, local validation of model/runtime identities, and retirement of stale validation receipts.
- User and troubleshooting guides, contribution/security policies, issue templates, dependency provenance, and automated Windows checks.

## Validation scope

The Windows checks install from a clean checkout with Python 3.12 and 3.13, run the regression suite, then create a synthetic edit with two frame rates, music, all three social previews, resume and music removal. Output checks include frame counts, timing, complete decoding, audio, and original-file hashes. The optional model test is intentionally skipped on hosts without downloaded models.

Local release checks also exercise a fresh folder containing spaces and brackets, deliberate package damage followed by repair, the first-run interface at smaller window sizes, real 4K CPU fallback, and bounded inference using the optional models. These checks establish the tested behavior; they do not certify every GPU/driver, artistic quality or named-trick accuracy.

## Read before a long render

Named trick recognition remains experimental. Review automatic labels, complete maneuvers and exits before posting. Internet-pretrained weights and published FPV definitions provide the model's knowledge; this release does not claim training on all online drone footage.

4K output preserves the selected canvas and timing, but scaling cannot create native camera detail. Inspect an AI sample for natural texture before restoring a long edit. HDR/log conversion, non-square-pixel normalization, gyro stabilization and synthetic slow motion are outside the current ingest path.

GPU inference, scaling and encoding are separate capabilities. FFmpeg's NVENC support needs a compatible driver; otherwise encoding uses the CPU and can take longer. Video understanding currently needs an NVIDIA GPU with at least 7 GiB of available device capacity; the broad scene model can also run on CPU. Setup does not change GPU drivers or system security settings.

## Source and support

The ZIP contains source, tests, documentation, notices and manifests. Footage, music, saved jobs, logs, environments, downloaded tools and model weights are excluded. **SHA256SUMS.txt** identifies the archive and **FPV-Sesh-0.4.0-source-manifest.json** records every source file and its commit.

Read the [user guide](user-guide.md), [troubleshooting](troubleshooting.md), and [third-party notices](third-party.md). Report reproducible problems through the repository's issue templates; follow [SECURITY.md](../SECURITY.md) for vulnerabilities.
