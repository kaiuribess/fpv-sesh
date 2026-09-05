"""Places365 scene inference; optional Torch runtime, never a trick classifier.

Checkpoint-compatible ResNet18 follows torchvision's BSD-3-Clause architecture.
See models/places365/Torchvision-BSD-LICENSE.txt and MODEL-LICENSE.md.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.downsample = (nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False),
                                       nn.BatchNorm2d(outgoing))
                           if stride != 1 or incoming != outgoing else None)
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + identity)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 365)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


SCENE_GROUPS = {
    "woodland": {"bamboo_forest", "forest/broadleaf", "forest_path", "forest_road", "rainforest", "tree_farm"},
    "park or open grass": {"park", "lawn", "pasture", "golf_course", "field/wild", "hayfield", "athletic_field/outdoor", "soccer_field", "football_field", "baseball_field"},
    "cultivated field": {"field/cultivated", "corn_field", "wheat_field", "orchard", "rice_paddy", "vineyard"},
    "sky": {"sky"},
    "built surroundings": {"building_facade", "residential_neighborhood", "apartment_building/outdoor", "office_building", "street", "parking_lot", "playground"},
    "water": {"ocean", "lake/natural", "river", "beach", "lagoon", "pond"},
}


class SceneModel:
    def __init__(self, folder=None, device=None):
        folder = Path(folder or ROOT / "models/places365")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["assets"]:
            path = folder / entry["file"]
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != entry["size_bytes"] or actual != entry["sha256"]:
                raise ValueError(f"Scene model asset integrity mismatch: {path.name}")
        self.labels = [line.split()[0][3:] for line in (folder / "categories_places365.txt").read_text().splitlines()]
        if len(self.labels) != 365:
            raise ValueError("Scene model requires its exact 365-category label order")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(2)
        model = ResNet18()
        checkpoint = torch.load(folder / "resnet18_places365.pth.tar", map_location="cpu", weights_only=True, encoding="latin1")
        state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(self.device)
        self.model_hash = manifest["assets"][0]["sha256"]

    def predict(self, bgr_frames):
        """Return top classes and grouped softmax mass, not calibrated confidence."""
        arrays = []
        for frame in bgr_frames:
            rgb = Image.fromarray(frame[:, :, ::-1]).resize((256, 256), Image.Resampling.BILINEAR).crop((16, 16, 240, 240))
            array = np.asarray(rgb, dtype=np.float32) / 255.0
            arrays.append(((array - [.485, .456, .406]) / [.229, .224, .225]).transpose(2, 0, 1))
        if not arrays:
            return []
        tensor = torch.from_numpy(np.stack(arrays).astype(np.float32)).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1).cpu().numpy()
        results = []
        for values in probabilities:
            top = np.argsort(values)[-5:][::-1]
            groups = {group: round(float(sum(values[i] for i, label in enumerate(self.labels) if label in names)), 5)
                      for group, names in SCENE_GROUPS.items()}
            group, score = max(groups.items(), key=lambda item: item[1])
            results.append({"top_classes": [{"label": self.labels[i], "score": round(float(values[i]), 5)} for i in top],
                            "groups": groups, "scene": group if score >= .35 else "uncertain scene",
                            "score": score, "method": "online-pretrained Places365 scene estimate"})
        return results
