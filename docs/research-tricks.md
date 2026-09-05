# Internet-pretrained trick and flight understanding

Research and setup review: 2026-09-05. The user's request is for knowledge learned from internet material that helps interpret future footage, including footage containing no deliberate tricks. Requiring a pilot to submit labeled tricks first would not satisfy that request.

The practical starting point is an existing video-language checkpoint combined with temporal motion evidence and a clear uncertain result. Downloading a pretrained checkpoint imports learned visual knowledge; reading tutorials can inform definitions and validation rules. These are different from training a new FPV classifier on a collection of internet videos. No such new training or video scraping is claimed here.

## Model decision

| Option | Verified facts | Decision for an RTX 4060 Ti with 8 GB |
| --- | --- | --- |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Apache-2.0, 2.128B BF16 parameters. Official model weights are 4.255 GB. Native Transformers support includes timestamp-aware video inputs and SDPA. | Installed and exercised on the local GPU. Its missed synthetic rotation demonstrates why independent motion evidence is necessary. |
| [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | Apache-2.0. Its card documents 3.3M multimodal training samples, 33% video, and a publisher-reported 1.8 GB video inference memory figure. | A smaller fallback to evaluate if Qwen is too slow or memory-constrained. It is not installed by this setup. |
| [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/tree/main) | About 8.99 GB of F32 weights on disk, usable at lower precision. | Larger download than Qwen; no demonstrated FPV advantage. Not selected. |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main) | About 7.52 GB of weights/configuration. Its actual [license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) is the Qwen Research License, restricted to noncommercial research/evaluation without a separate commercial license. | Too little memory headroom for the default and a less suitable distribution license. Do not assume every Qwen checkpoint is Apache-2.0. |
| [CameraBench camera-motion model](https://huggingface.co/chancharikm/qwen2.5-vl-7b-cam-motion) | A real camera-motion-specialized Qwen2.5-VL checkpoint, trained using 8 FPS. Released versions start at 7B. | Useful evidence that specialized training improves camera-motion understanding. Standard BF16 weights exceed this GPU's memory; quantized/offloaded deployment is additional work. Not installed. |

The Qwen3-VL publisher reports improved temporal modeling, but there is no verified head-to-head FPV trick benchmark proving its advantage over SmolVLM. Selection is an engineering judgment based on capability, practical size and licensing. [Official Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL).

## Local input and runtime contract

Use native `Qwen3VLForConditionalGeneration` and `AutoProcessor`, BF16 weights, `attn_implementation="sdpa"`, `local_files_only=True` and `trust_remote_code=False`. The release runtime is Transformers 5.10.2 with Torch 2.14.0/CUDA12.6; the initial evaluation used Transformers 4.57.6. The model uses 16-pixel patches, two-by-two spatial merging and two-frame temporal merging. Both dimensions should therefore be compatible with multiples of 32, and clips should use an even frame count. [Transformers Qwen3-VL documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl).

Decode selected frames locally and pass an RGB array shaped `[frames, height, width, channels]`. Pre-sampled video must include `video_metadata` with `total_num_frames`, `fps` and `frames_indices`; otherwise the processor may silently invent 24 FPS timestamps. Disable further frame sampling. Frame indices and FPS must describe the actual sampling time base, including variable-rate footage handled through explicit timestamps or a known constant-rate proxy.

The video processor's `size["longest_edge"]` is a total `T * H * W` pixel budget, unlike the image processor's per-image budget. The implemented profiles use the following limits, one window per batch. [Official processor controls](https://github.com/QwenLM/Qwen3-VL#pixel-control-via-official-processor).

| Profile | Window and sampling | Full-window model canvas | Total pixel budget |
| --- | --- | --- | --- |
| Automatic | Up to 8 seconds at 8 FPS; at most 64 frames | 128 × 128 per frame, about 512 visual tokens | 1,048,576 |
| Thorough | Up to 6 seconds at 16 FPS; at most 96 frames | 192 × 192 per frame, about 1,728 visual tokens | 3,538,944 |

The local decoder preserves the original aspect ratio inside a square before model resizing. Windows overlap to provide context around transitions; shorter boundary windows use fewer sampled frames. These samples can still miss very fast rotations or fine detail. Thorough spends more time and memory on denser temporal and spatial input; it is not a promise of correct trick names.

For a predecoded constant-rate sample, a minimal processor call is:

```python
text = processor.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": question},
    ]}], tokenize=False, add_generation_prompt=True,
)
inputs = processor(
    text=[text], videos=[rgb_frames],
    video_metadata=[{
        "total_num_frames": len(rgb_frames),
        "fps": sample_fps,
        "frames_indices": list(range(len(rgb_frames))),
    }],
    do_sample_frames=False, return_tensors="pt",
)
```

This metadata describes time relative to the sampled clip. Keep the original source start separately when mapping results back to a flight. Do not fabricate FPS merely to satisfy the API.

## Evidence, uncertainty and complete moments

Separate basic motion from named tricks. Full rolls need evidence of a complete lateral rotation; flips need a complete pitch rotation; multiple flips need distinct complete cycles. A powerloop around an obstacle needs the expected trajectory and obstacle relationship. A split-S combines inversion and a descending half-loop; an obstacle is not required. Context such as trees can help interpret a weaving line, but cannot establish its motion by itself. The [IGOW competition's own challenges](https://sites.google.com/view/igow/challenges) distinguish named maneuvers, obstacle relationships and continuous lines; these inform the recognition questions, not training labels automatically harvested from a webpage.

The implemented vocabulary also draws on [Rotor Riot's FPV Freestyle Tricktionary](https://rotorriot.com/blogs/tutorials-guides/fpv-freestyle-tricktionary), which separates rotation families, obstacle maneuvers and intentional contacts. [ArduPilot's flip-mode documentation](https://ardupilot.org/copter/docs/flip-mode.html) distinguishes roll/pitch rotation from returning to the entry attitude. These references supply definitions and checks; their existence does not establish that Qwen recognizes the corresponding flight sequence.

Include ordinary flight and uncertain outcomes in every classification prompt. Require observable evidence, check claimed rotations against measured motion, and retain the raw model response for review. Generated confidence numbers and next-token probabilities are not calibrated percentages of trick correctness. A known negative clip is essential when testing: the system must not invent tricks just because the prompt lists trick names.

Keep lead-in and recovery protected regardless of the suggested name. A model recognizing an action midway through a clip must not become a reason to cut immediately after a second flip. Scene labels must not imply a successful landing, airborne state or crash-free completion.

This conservatism is evidence-based: [CameraBench](https://linzhiqiu.github.io/papers/camerabench/) finds video-language models weak at exact geometric motion despite useful scene semantics. [ChaChaBench](https://github.com/carihkl/ChaChaBench) reports substantial confusion between rotations and translations, even in simplified camera-motion clips. Neither benchmark establishes accuracy for FPV freestyle.

## Actual local evaluation and its limits

The Automatic profile was exercised on the RTX 4060 Ti using two ordinary-looking real flight windows and two controlled synthetic image transforms. The four evaluation records took 19.3–20.6 seconds per eight-second window, excluding initial model loading. Each recorded 4,525 MiB of peak PyTorch allocated GPU memory and 4,574 MiB reserved. These are allocator measurements, not total desktop/GPU usage, and do not establish Thorough-profile performance.

A separate Thorough smoke check decoded 96 frames from the first six seconds of the same synthetic rotation control. It completed in 60.3 seconds with 5,533 MiB peak PyTorch allocation and 5,680 MiB reserved. The video model again missed the rotation; the independent tracker supplied the possible-roll suggestion. This verifies that the larger profile runs on this GPU, without establishing better recognition accuracy.

| Input | Observed result | What this establishes |
| --- | --- | --- |
| DJIU0001, 40–48 seconds | Ordinary flight; no complete image rotation measured | A real banking/park-flight example did not receive a named trick suggestion. |
| DJIU0002, 30–38 seconds | Ordinary flight; no complete image rotation measured | A second real flight window did not receive a named trick suggestion. |
| Frozen source frame with a synthetic ±20° bank | Ordinary flight category; no complete image rotation measured | The bounded bank did not trigger a roll suggestion. The model's prose nevertheless invented forward flight from a rotating still image, so descriptive text is not independently verified. |
| Frozen source frame with a synthetic 360° rotation during seconds 2–4 | Qwen alone returned ordinary flight and missed the full rotation. Independent 30 FPS tracking measured a continuous 349.33° image rotation and supplied a possible-roll suggestion. | The image-rotation measurement can catch a clear temporal event missed by the model. The result is explicitly attributed to measured image rotation with pretrained video context. |

The synthetic controls are eight-second, 256 × 256, 30 FPS clips with 240 decoded frames. They rotate a photograph; they do not demonstrate a physical drone roll, pitch axis, translational flight path, obstacle relationship or crash-free recovery. The rotation control has two seconds of unchanged entry, two seconds of added rotation and four seconds of unchanged exit. Quieter image motion afterward is not evidence of airborne recovery.

The first moving-source transform pair retained unrelated underlying camera motion and was therefore not treated as a clean negative/positive trick benchmark. The controlled still pair isolates the added transform. Exact filters, hashes, source times, readable contact sheets and raw observations are recorded locally under `logs/video-evaluation/`; originals were hash-checked and stayed unchanged. These evaluation assets remain outside Git and were not used for training.

The two real examples were selected from ordinary-looking flight and visually inspected through sampled contact sheets. This small, nonrandom set is not a labeled FPV accuracy benchmark. Tracking was partial on the first real window, so absence of a measured rotation cannot prove absence of a trick. No precision/recall percentage is reported. Recognition of real flips, double flips, powerloops, split-S maneuvers and complete recoveries remains unvalidated. Internet pretraining is present, but it does not justify presenting these suggestions as verified trick detection.

The completed Automatic scan of all three supplied recordings covered 83 overlapping windows and 567.467 seconds of source time. It returned 82 ordinary-flight observations and one uncertain observation caused by an unusable structured response. These broad labels do not establish that weaving or other maneuvers were absent. The first complete scan took about 27 minutes; a repeat refresh reused all observations, loaded no video model, and completed in seconds. Both refreshes preserved all candidate edit data and the exact hashes of 13 existing edit/export files, including the eight-shot timeline and rendered videos. This is an integration and coverage check, not a trick-accuracy benchmark.

## FPV datasets examined

| Source | What its labels actually represent | Suitability |
| --- | --- | --- |
| [UZH-FPV](https://fpv.ifi.uzh.ch/) | Aggressive drone racing footage, images, IMU and ground-truth trajectories; CC BY-NC-SA 3.0. The SplitS track name is a track identifier, not a frame-level freestyle trick annotation. | Relevant to motion estimation research. It is not a ready labeled flips/rolls/powerloops training set. Not downloaded. |
| [Imageomics drone-maneuver-clips](https://huggingface.co/datasets/imageomics/drone-maneuver-clips) | Wildlife drone footage and controller actions generated by a navigation decision tree; compilation CC BY 4.0. | Proposed navigation actions are not observations of flown freestyle tricks. Not downloaded or used as trick ground truth. |
| [Deep Drone Acrobatics](https://github.com/uzh-rpg/deep_drone_acrobatics) | A research control and simulation project for executing acrobatics. | Relevant demonstrations and control research, but not a drop-in FPV video trick classifier. Not downloaded. |
| [CameraBench](https://github.com/sy77777en/CameraBench) | Approximately 3,000 internet clips with expert camera-motion annotations; public test set and trained models, with training access requested separately. | Closest examined specialist resource, covering primitives rather than an FPV freestyle taxonomy. Not downloaded. |

These searches did not locate a verified openly downloadable dataset pairing real first-person FPV freestyle footage with reliable time-bounded labels for all requested tricks. This is a limited search finding, not a claim that no such dataset exists. Future FPV-specific fine-tuning needs a rights-cleared corpus, expert annotations, held-out pilots/locations and a test set with many ordinary-flight negatives. It should not be described as accomplished by installing a general model.
