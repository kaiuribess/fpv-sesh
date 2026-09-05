# Contributing to FPV Sesh

Start with the [user guide](docs/user-guide.md) and [troubleshooting](docs/troubleshooting.md). Use an issue to explain a reproducible bug or a concrete improvement. Small, focused pull requests are easier to review.

## Local development

Use a separate checkout and the supported Python version from [README](README.md). Run `install.cmd -Development` to include pytest and the development tools, then work from the repository's own environment. The default core installation excludes development packages. Core editing and its tests must work without optional model downloads.

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m fpvsesh.cli make --help
```

Run tests relevant to the change before the full suite. Media tests generate their own temporary recordings and may need the configured FFmpeg installation. GUI checks require Windows with Tkinter. Optional GPU/model checks are separate: record them explicitly rather than treating skipped model tests as proof of working inference. Run `doctor.cmd` to inspect the selected runtime and tools.

## Changes worth preserving

- Keep original recordings read-only. Saved source identities and exact user-reviewed intervals must remain meaningful after resume or regeneration.
- Keep missing optional models compatible with ordinary editing. State the actual encoder, scaling and inference path instead of inferring readiness from a GPU name.
- Treat file paths, media metadata, model text and saved job records as data. Preserve argument-list subprocess calls, bounded decoding and recorded asset verification.
- Keep predicted tricks, measured motion and user confirmations distinct. Do not add accuracy percentages without a defined independent benchmark.
- Keep pause/cancel checkpoints responsive and retain only complete, verified cache entries for reuse.

## Tests and public examples

Use generated media for regression tests and public screenshots. Fixtures must not reveal a user's home, property, location, username, account, recordings or music. Never commit local jobs, environments, checkpoints, caches, logs, credentials or private media. Do not force-add ignored files to make a test pass.

For a rendering change, check output dimensions, timing, complete decode and the behavior the change is intended to protect. For UI changes, exercise keyboard operation and inspect both default and minimum window sizes. A still image cannot establish temporal video quality; describe what was actually checked.

## Dependencies and attribution

Keep exact versions and publisher hashes aligned with setup manifests. Review public upstream advisories, compatibility and wheel availability before changing a pin. Test installation in an empty application copy; success in a pre-populated environment does not prove a fresh install works. Do not weaken hash checks or load remote model code to work around an installation error.

Retain upstream license notices and identify adapted code. Source releases contain the application and notices; external tools and models download separately. See [third-party notes](docs/third-party.md).

## Pull requests

Explain the problem, resulting behavior and relevant verification. Include limitations or remaining checks. Screenshots and logs should be redacted and reproducible; synthetic footage is preferred. Contributions to FPV Sesh's own code are under the repository's MIT license; third-party components retain their upstream terms.

Report vulnerabilities through [SECURITY.md](SECURITY.md), not a public exploit demonstration. Be respectful and specific in reviews, and discuss the behavior rather than the person proposing it.
