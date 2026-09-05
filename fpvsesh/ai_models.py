"""Small CUDA-only inference adapter for official Real-ESRGAN checkpoints.

Architecture portions adapted from Xintao Wang's Real-ESRGAN (2021,
BSD-3-Clause) and BasicSR Authors' RRDBNet (2018-2022, Apache-2.0).
Full licenses and exact source URLs are in models/real-esrgan-cuda/.
Changes: remove training/registry dependencies, add safe checkpoint loading,
bounded CUDA OOM retries, CPU tile assembly and per-frame diagnostics.
This module does not download models, install dependencies or alter inputs.
"""

from __future__ import annotations

import gc
import hashlib
import math
import time
from pathlib import Path

import cv2
import numpy as np
from .runtime_dlls import prepare_torch_dlls

prepare_torch_dlls()
import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Checkpoint-compatible RealESRGAN_x2plus architecture."""

    def __init__(self):
        super().__init__()
        self.scale = 2
        self.conv_first = nn.Conv2d(12, 64, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB() for _ in range(23)))
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(F.pixel_unshuffle(x, 2))
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """Checkpoint-compatible general-x4v3, 32 inner convolutions."""

    def __init__(self):
        super().__init__()
        self.body = nn.ModuleList([nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64)])
        for _ in range(32):
            self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
            self.body.append(nn.PReLU(64))
        self.body.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(4)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=4, mode="nearest")


def _checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("params_ema", payload.get("params")) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError(f"No model parameter mapping in {path.name}")
    if any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
           for value in state.values()):
        raise ValueError(f"Invalid or non-finite model parameters in {path.name}")
    return state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Restorer:
    """Persistent FP16 CUDA model with bounded per-frame tiling.

    ``enhance`` returns BGR uint8, with last_stats and metadata available.
    Denoise is fixed model-weight interpolation, never per-frame adaptation.
    Native x4 inference is labelled explicitly even when outscale=2 requests
    subsequent Lanczos reduction. No extra sharpening or source blend is added.
    """

    def __init__(self, model_name, model_dir, tile=384, denoise=0.2, tile_pad=40,
                 cudnn_benchmark=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch and an available NVIDIA GPU are required for AI enhancement")
        if model_name not in ("RealESRGAN_x2plus", "realesr-general-x4v3"):
            raise ValueError(f"Unsupported realistic model: {model_name}")
        if not 0 <= float(denoise) <= 1:
            raise ValueError("denoise must be between 0 and 1")
        if int(tile) < 128 or int(tile) > 1024 or int(tile) % 2:
            raise ValueError("tile must be an even integer from 128 through 1024")
        if int(tile_pad) < 40 or int(tile_pad) % 2:
            raise ValueError("tile_pad must be even and at least 40 input pixels")
        # Opt-in tuning must happen before the first inference for these shapes.
        # It costs a warm-up, but tile768 gained ~7% in the recorded long-run test.
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
            if cudnn_benchmark and hasattr(torch.backends.cudnn, "benchmark_limit"):
                torch.backends.cudnn.benchmark_limit = 8
        started = time.perf_counter()
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.tile = int(tile)
        self.tile_pad = int(tile_pad)
        self.denoise = float(denoise)
        self.device = torch.device("cuda:0")
        self.scale = 2 if model_name == "RealESRGAN_x2plus" else 4
        paths = [self.model_dir / (model_name + ".pth")]
        state = _checkpoint(paths[0])
        if self.scale == 4 and self.denoise != 1.0:
            paths.append(self.model_dir / "realesr-general-wdn-x4v3.pth")
            weak = _checkpoint(paths[1])
            if state.keys() != weak.keys():
                raise ValueError("Denoise checkpoint keys differ")
            state = {key: value * self.denoise + weak[key] * (1.0 - self.denoise)
                     for key, value in state.items()}
            del weak
        model = RRDBNet() if self.scale == 2 else SRVGGNetCompact()
        model.load_state_dict(state, strict=True)
        del state
        self.model = model.eval().requires_grad_(False).to(
            device=self.device, dtype=torch.float16, memory_format=torch.channels_last)
        torch.cuda.synchronize(self.device)
        self.metadata = {
            "backend": "pytorch-cuda", "version": "fpv-real-esrgan-cuda-v1",
            "model_name": self.model_name, "native_scale": self.scale,
            "denoise": self.denoise if self.scale == 4 else None,
            "precision": "float16", "memory_format": "channels_last",
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_benchmark_limit": getattr(torch.backends.cudnn, "benchmark_limit", None),
            "tile_requested": self.tile, "tile_pad": self.tile_pad,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(self.device),
            "checkpoints": [{"name": p.name, "sha256": _sha256(p)} for p in paths],
            "load_seconds": time.perf_counter() - started,
        }
        self.last_stats = {}
        self.frames_processed = 0

    @torch.inference_mode()
    def _tiles(self, bgr, tile):
        height, width = bgr.shape[:2]
        # Even padded dimensions preserve pixel-unshuffle phase for native x2.
        if self.scale == 2 and (height % 2 or width % 2):
            bgr = cv2.copyMakeBorder(bgr, 0, height % 2, 0, width % 2, cv2.BORDER_REFLECT_101)
        padded_h, padded_w = bgr.shape[:2]
        result = np.empty((padded_h * self.scale, padded_w * self.scale, 3), dtype=np.uint8)
        tiles_count = 0
        for top in range(0, padded_h, tile):
            bottom = min(top + tile, padded_h)
            for left in range(0, padded_w, tile):
                right = min(left + tile, padded_w)
                x0, y0 = max(0, left - self.tile_pad), max(0, top - self.tile_pad)
                x1, y1 = min(padded_w, right + self.tile_pad), min(padded_h, bottom + self.tile_pad)
                rgb = np.ascontiguousarray(bgr[y0:y1, x0:x1, ::-1].transpose(2, 0, 1))
                tensor = torch.from_numpy(rgb).unsqueeze(0).to(
                    device=self.device, dtype=torch.float16, memory_format=torch.channels_last)
                tensor = tensor / 255.0
                restored = self.model(tensor)
                s = self.scale
                restored = restored[0, :, (top-y0)*s:(bottom-y0)*s, (left-x0)*s:(right-x0)*s]
                # Fail instead of silently converting invalid CUDA results to black.
                if not bool(torch.isfinite(restored).all()):
                    raise RuntimeError("AI produced non-finite pixels")
                restored = restored.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
                restored = restored.permute(1, 2, 0).cpu().numpy()
                result[top*s:bottom*s, left*s:right*s] = restored[:, :, ::-1]
                del tensor, restored
                tiles_count += 1
        return result[:height*self.scale, :width*self.scale], tiles_count

    def enhance(self, bgr_uint8, outscale=2):
        if not isinstance(bgr_uint8, np.ndarray) or bgr_uint8.dtype != np.uint8:
            raise ValueError("enhance expects a uint8 NumPy BGR frame")
        if bgr_uint8.ndim != 3 or bgr_uint8.shape[2] != 3 or min(bgr_uint8.shape[:2]) < 1:
            raise ValueError("enhance expects a nonempty H×W×3 BGR frame")
        if not math.isfinite(float(outscale)) or not 0 < float(outscale) <= self.scale:
            raise ValueError("outscale must be positive and at most the native model scale")
        started = time.perf_counter()
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        candidates = list(dict.fromkeys([self.tile, min(self.tile, 256), 128]))
        retries = 0
        for tile in candidates:
            try:
                output, tiles_count = self._tiles(bgr_uint8, tile)
                self.tile = tile
                break
            except torch.cuda.OutOfMemoryError:
                retries += 1
                gc.collect()
                torch.cuda.empty_cache()
                if tile == candidates[-1]:
                    raise RuntimeError("AI exhausted CUDA memory even at 128-pixel tiles") from None
        height, width = bgr_uint8.shape[:2]
        if float(outscale) != self.scale:
            output = cv2.resize(output, (max(1, int(width * outscale)), max(1, int(height * outscale))),
                                interpolation=cv2.INTER_LANCZOS4)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.frames_processed += 1
        self.last_stats = {
            "elapsed_seconds": elapsed, "fps": 1.0 / elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "tile": self.tile, "tile_pad": self.tile_pad, "tiles": tiles_count,
            "oom_retries": retries, "frames_processed": self.frames_processed,
            "native_scale": self.scale, "outscale": float(outscale),
            "input_width": width, "input_height": height,
            "output_width": output.shape[1], "output_height": output.shape[0],
        }
        return output

    def close(self):
        """Release model allocations explicitly before testing another model."""
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
