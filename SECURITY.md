# Security policy

FPV Sesh is a local desktop application that processes media and downloads explicitly selected dependencies. It is not a sandbox for hostile media, model files or saved jobs. Use release files and the recorded setup sources, and keep supported dependencies current.

## Report a vulnerability

Use **Security → Report a vulnerability** on the [repository security page](https://github.com/kaiuribess/fpv-sesh/security) when private reporting is available. Include affected release/commit, impact, the relevant component and a minimal reproduction using synthetic data. Do not include live credentials or another person's footage.

If private reporting is unavailable, open an issue containing only a request for a private security contact. Do not include exploit details or sensitive files in that public request. No response-time guarantee or paid support agreement is offered.

## Supported scope

Security fixes target the current development branch and the latest release. Earlier releases and locally modified dependency stacks may require an upgrade. Optional models have separate dependencies and compatibility limits; installing them is not required for core editing. Published package advisories and their actual reachability should both be assessed before a release.

## Data and network behavior

Ordinary analysis, editing and installed-model inference run locally. Setup contacts package, tool and model publishers to download selected assets. The app does not require an API key, upload footage for inference or publish videos to social accounts.

Saved jobs, diagnostic reports, logs and posters can contain full source paths, machine information and recognizable footage. Review and redact them before sharing. Git ignores these local outputs; an ignore rule does not remove files that were previously committed, so release preparation must inspect the actual archive and reachable history.

## Maintainer release checks

- Verify exact dependency versions, upstream advisories, hashes and license notices.
- Build the source ZIP from an explicit source inventory; do not archive a working folder wholesale.
- Inspect archive contents and history for credentials, media, personal paths and local diagnostics.
- Test a clean installation and core editing without optional models.
- Keep model configuration and checkpoint verification in place; do not enable remote Python model code or unsafe pickle fallback.
- Recheck the exact release commit after dependency or packaging changes. Passing tests or an advisory scan is not a security guarantee.
