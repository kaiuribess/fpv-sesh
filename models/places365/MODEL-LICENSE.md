# Places365 attribution and provenance

The pretrained ResNet18 weights are provided by the MIT CSAIL Places project under **Creative Common License (Attribution CC BY)**, as stated in its [official README](https://github.com/CSAILVision/places365#acknowledgements-and-license). Upstream does not identify a CC BY version; this project does not substitute or invent one. The images used by the authors remain owned by their respective owners.

Credit: Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, Antonio Torralba. *Places: A 10 million Image Database for Scene Recognition*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017. [Places project](http://places2.csail.mit.edu/).

The upstream Places365 **code** is MIT licensed; its complete notice is preserved in `Places365-MIT-LICENSE.txt`. The checkpoint-compatible ResNet18 architecture follows torchvision, whose complete BSD-3-Clause notice is preserved in `Torchvision-BSD-LICENSE.txt`. Those code licenses do not replace the pretrained weight license.

The exact source URLs, sizes, and complete SHA256 hashes are recorded in `manifest.json`. The official model download uses HTTP; the host rejected HTTPS during setup. The digest was calculated after the official download, and no independent publisher-signed digest was found. The pinned digest verifies subsequent copies against that initial download; it is not a claim of cryptographic publisher authentication. PyTorch loads the checkpoint with `weights_only=True`; executable pickle fallback is never used.

Only code, metadata, category labels, and notices belong in Git. Downloaded weights and scene caches remain local. No source footage is sent to the model authors or any online service.
