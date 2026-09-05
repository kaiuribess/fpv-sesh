# FPV Sesh research and design decisions

Reviewed 5 September 2026. This is a targeted review of representative FPV production work, public datasets, and primary technical sources. It is not an exhaustive survey of every online flight, and linked films were not downloaded as a training corpus. Implementation and measured results are separate from research claims.

## Preserve the flight before decorating the edit

Red Bull's [One Shot with Johnny FPV](https://www.redbull.com/us-en/one-shot-with-johnny-schaer) describes a continuous action-following production. GoPro's [Kilian Bron and Pierre Dupont interview](https://gopro.com/en/qa/news/million-dollar-challenge-creators-kilian-bron-pierre-dupont-) discusses deliberate planning, repeated takes, and a complete trick capture. These examples support an editorial inference: approaches, spatial relationships, and exits often carry the excitement in FPV footage. Rapid cuts are an option, not a universal improvement.

FPV Sesh therefore offers energetic, cinematic, freestyle, and longer continuous-line styles. Story order builds a reel; chronological order keeps selected passages in input-file order and source time. A configurable recovery interval links motion bursts and protects follow-through. Explicitly reviewed intervals take precedence over automatic timing.

## Music follows the flying

Beat detection measures accents in a waveform. It does not tell the editor whether a maneuver is finished. Our policy extends an automatic exit by at most 0.8 seconds toward a clear detected beat, only if source limits, selected-interval overlap, landing context, duration limits, and recovery checks permit it. It never shortens a passage to hit a beat. Reviewed and explicitly kept cuts stay exact. Some cuts intentionally remain off-beat.

The app accepts local music, supports track offsets and independent music/flight-sound levels, fades the available music or repeats it explicitly, and verifies the final decoded soundtrack. It uses a lossless source-sound intermediate before mixing. See [music research](research-music.md) for the primary signal-processing sources, implementation details, and test scope.

## Each social layout starts with the original footage

[YouTube's upload guidance](https://support.google.com/youtube/answer/1722171?hl=en) supports progressive video, the recorded frame rate, suitable resolution/bitrate, and common upload containers. [YouTube's Shorts guide](https://support.google.com/youtube/answer/15424877?hl=en) classifies square or vertical videos up to three minutes as Shorts for standard channels. Platform behavior can change after upload, so the exporter reports format facts rather than guaranteeing reach or playback quality.

The landscape master remains 3840×2160. Social versions are 1080×1920, 1080×1080, and 1080×1350, rendered from the original source intervals. The default blurred background preserves the entire flight view; fit uses bars; fill deliberately crops with a user-selected horizontal focus. Cropping is never presented as reliable automatic subject tracking. See [social research](research-social.md) for current Meta/TikTok documentation, encoding choices, and verification.

## Natural detail and stabilization

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) provides practical pretrained restoration models, including native 2× and tiled inference. A model's sharper result is not evidence that it recovered the true scene. Research such as [VideoGigaGAN](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_VideoGigaGAN_Towards_Detail-rich_Video_Super-Resolution_CVPR_2025_paper.pdf) explicitly studies the tension between fine detail and temporal consistency. Its results do not validate this application's independent frame-based model.

The default here is natural color with zero added grade and the tested conventional upscale. Optional CUDA restoration remains a restrained blend, with an actual local inference/decode test required before activation. A sample test establishes that a backend works; the user still judges foliage texture, shimmer, and motion. Neither 4K dimensions nor 10-bit output can create missing camera information.

[Gyroflow](https://github.com/gyroflow/gyroflow) is a dedicated stabilization system using motion data and lens information. GoPro's [ReelSteady explanation](https://gopro.com/en/rs/news/hypersmooth-reelsteady-technology-explained) also describes camera metadata as part of its stabilization approach. FPV Sesh does not silently level intentional rolls or pretend that optical flow alone supplies synchronized gyro data. A validated stabilization integration remains separate work.

## What online data can actually teach

The [UZH-FPV dataset](https://fpv.ifi.uzh.ch/) supplies aggressive racing sequences, images, inertial measurements, and trajectory ground truth. It is valuable for visual-inertial state estimation, but its published task is not a labeled freestyle-trick taxonomy. Its noncommercial share-alike terms also need to be preserved for any dataset use.

[Deep Drone Acrobatics](https://github.com/uzh-rpg/deep_drone_acrobatics) learns to control acrobatic flight using simulation and privileged training information. A controller trained to execute a loop is not automatically a classifier that recognizes a loop in arbitrary edited video.

The [Imageomics drone-maneuver clip library](https://huggingface.co/datasets/imageomics/drone-maneuver-clips) contains wildlife filming clips. Its action labels are generated by replaying a controller policy over animal annotations; they should not be mistaken for measured camera motion or FPV stunt labels. It was rejected for training freestyle recognition here.

[MIT Places365](https://github.com/CSAILVision/places365) releases scene models trained on approximately 1.8 million images across 365 scene categories. Such pretrained visual features can help distinguish surroundings, while temporal motion features describe how the camera is moving. Scene classification alone cannot prove a roll, double flip, powerloop, crash, or a metric 3D route. See the online-model notes for what was actually integrated and tested.

## Remaining work requires evidence

Reliable named-trick recognition needs representative, properly labeled FPV sequences, independent pilots/flights in evaluation, and explicit abstention on ambiguous examples. Video-only geographic mapping additionally needs a validated reconstruction pipeline and a way to resolve scale. Neither is claimed by the current motion/scene timeline. More effects, stronger sharpening, and a longer feature list are not substitutes for accurate cuts and trustworthy outputs.
