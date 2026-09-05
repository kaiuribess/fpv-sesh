# Social export formats and verification

Checked 5 September 2026 against the primary sources below. These are local export profiles, not remote publishing or a guarantee that every account and upload surface accepts identical limits.

| Export | Final pixels | Preview pixels | Intended use |
|---|---|---|---|
| vertical | 1080×1920, 9:16 | 360×640 | Shorts, Reels, TikTok |
| square | 1080×1080, 1:1 | 360×360 | Square feeds; square YouTube Shorts |
| portrait | 1080×1350, 4:5 | 360×450 | Portrait feeds; taller-than-square YouTube Shorts |

The separate YouTube master remains 3840×2160. Social exports are labelled by their actual dimensions and are rendered from the original selected source intervals. They never decode and re-encode the pillarboxed 4K master.

## Primary platform findings

YouTube recommends MP4, H.264 High Profile, 4:2:0, AAC audio at 48 kHz, front-loaded MP4 metadata, original frame rate, and BT.709 for SDR. The player adapts to vertical and square aspect ratios. Its recommended 1080p high-frame-rate upload rate is 12 Mb/s; this is guidance rather than a bitrate limit. [YouTube encoding guidance](https://support.google.com/youtube/answer/1722171?hl=en)

YouTube classifies square or vertical uploads up to three minutes as Shorts under its current rules. The exporter warns when a timeline exceeds 180 seconds and preserves it intact rather than cutting a trick to fit. [YouTube Shorts duration guidance](https://support.google.com/youtube/answer/15424877?hl=en)

Meta's own Instagram API collection lists MP4/MOV, H.264/HEVC, AAC 48 kHz, 23–60 fps, recommended 9:16, maximum 1920 horizontal pixels, 25 Mb/s video, 128 kb/s audio, 3 seconds–15 minutes, and 1 GB for its Reels publishing endpoint. These API specifications are distinct from manual upload and advertising surfaces. Our shared social encode uses H.264 8-bit, 20 Mb/s maximum-rate control, and AAC 128 kb/s at 48 kHz. Existing canonical AAC is copied only when its rate, channels and timing are compatible; otherwise the canonical track is normalized for these exports. [Meta's Instagram API documentation](https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api)

Facebook announced that its unified video/Reels publishing experience removes the previous length and format restrictions as it rolls out. A universal hard 90-second Facebook cutoff would therefore be incorrect. We apply no such cutoff. [Meta announcement](https://about.fb.com/news/2025/06/making-it-easier-create-videos-facebook/)

TikTok's Content Posting API recommends MP4/H.264, allows 23–60 fps, requires each picture dimension to be 360–4096 pixels, and documents a 4 GB maximum. Maximum publishing duration varies by account; the API can accept uploads up to 10 minutes for subsequent user trimming. We warn about frame rates outside that documented range and preserve the selected frame rate. [TikTok media transfer guide](https://developers.tiktok.com/docs/en/content-posting-api-media-transfer-guide)

## Framing and fidelity

- **Blur (default):** the full original view remains centered. A small, blurred and darkened copy fills the background. Tree approaches, lateral motion and recoveries remain visible in the foreground.
- **Fit:** the full view is contained with intentional black bars.
- **Fill:** the view covers the output canvas and crops its edges. `focus_x` is a fixed horizontal position from 0 (left) to 1 (right), default 0.5. This is an explicit crop choice, not subject tracking. The plan warns that edge details may disappear.

The graph uses FFmpeg's documented `split`, `scale`, `setsar`, `pad`, `crop`, `gblur`, `overlay`, `fps`, `trim`, and timestamp-reset behavior. Aspect calculations include sample aspect ratio and display rotation; output pixels are square. The selected fixed shot grade is reused from the main renderer. No neural inference or invented detail is claimed for social resizing. [FFmpeg7.1.1 filter documentation](https://github.com/FFmpeg/FFmpeg/blob/n7.1.1/doc/filters.texi)

Each original interval is encoded once per profile, with exact timeline frame count and chosen rational rate. Completed segments are stream-copied together. Canonical source/music audio comes from `job/source-audio.m4a`; when absent, the export contains a silent AAC track. This module never adds music. Final exports try NVENC H.264 P7/CQ17 and fall back to x264 medium/CRF17; lightweight previews use x264 fast/CRF18. Both use a 20 Mb/s video maximum-rate setting and 40 Mb buffer.

Cache keys include original identity and file timestamp, selected source time/frame bounds, output frame count/rate, profile dimensions, framing/focus, grade, source SAR/rotation, preview/encoder policy, renderer version and FFmpeg identity. Completed cache entries have complete-file SHA256 records. A changed cache file is regenerated. Unique temporary files are atomically promoted; cancellation removes only those temporary files. Original source files are checked for unexpected size/timestamp changes.

## API

`social_export_plan(settings, timeline=None, preview=False)` returns a dictionary keyed by requested format code. Values include dimensions, aspect ratio, framing, upload-oriented settings, platform notes and warnings. `settings['social_formats']` is a list; an empty list produces no exports.

`export_social(timeline, probes, settings, job, cache, event, checkpoint, preview=False)` returns:

```python
{
    "vertical": {
        "path": ".../social/vertical.mp4",
        "verification": {...},
        "records": [...],
        "profile": {...},
        "poster": ".../social/vertical-poster.jpg",  # None if optional extraction fails
        "warnings": [...],
    }
}
```

Preview files live under `job/social-preview/`; final files under `job/social/`. Each profile writes verification, backend and checkpoint JSON beside its output. The optional JPEG is taken from within the chosen opener. No upload, account login, external message or publishing action occurs.

## Tests and limits

All eight tests in `tests/test_social.py` passed in 17.2 seconds. They create real motion sources, including Unicode and shell-special filenames, and decode all three aspect ratios with all three framing modes. Checks cover complete frame counts, fractional 30000/1001 fps timing, AAC 48 kHz, faststart, full-frame black gaps, expected framing bars, focus selection and preservation of both source edges in blur mode. They also verify 1080×1920 final output, NVENC failure with a real software fallback encode, cache reuse/tamper recovery, cancellation cleanup, source immutability and 90-degree rotation with anamorphic pixels.

The framing tests use colored motion fixtures, so their geometry and timing assertions are meaningful. They do not prove the aesthetic quality of actual FPV footage or whether an automatically chosen interval contains a complete maneuver. Editorial boundaries remain the timeline's responsibility. Upload services may recompress files and apply account-specific restrictions; the module reports known mismatches instead of silently changing timeline length or frame rate.
