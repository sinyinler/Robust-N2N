"""SIDD RGB checkpoint、指标、tiled inference 与可视化公共函数。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from models.sidd_rgb_denoiser import SIDDRGBDenoiser
from utils.checkpoint import unwrap_state_dict


def load_sidd_model(checkpoint: str | Path, device: torch.device) -> SIDDRGBDenoiser:
    model = SIDDRGBDenoiser().to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = unwrap_state_dict(payload)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def psnr_rgb(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = float(np.mean((prediction - target) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def ssim_rgb(prediction: np.ndarray, target: np.ndarray) -> float:
    """接近 MATLAB/Wang SSIM 的 11x11 Gaussian RGB 局部窗口实现。"""

    return float(
        structural_similarity(
            np.asarray(prediction, dtype=np.float32),
            np.asarray(target, dtype=np.float32),
            channel_axis=-1,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            win_size=11,
        )
    )


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.inference_mode()
def tiled_inference(
    model: torch.nn.Module,
    image_u8: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
    amp: bool = True,
) -> np.ndarray:
    """对 HWC RGB uint8 大图重叠分块推理，余弦权重融合去除接缝。"""

    if image_u8.ndim != 3 or image_u8.shape[2] != 3:
        raise ValueError(f"期望 HWC RGB，得到 {image_u8.shape}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap 必须满足 0 <= overlap < tile_size")
    height, width = image_u8.shape[:2]
    y_starts = _tile_starts(height, tile_size, overlap)
    x_starts = _tile_starts(width, tile_size, overlap)
    output = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width, 1), dtype=np.float32)

    window_1d = np.hanning(tile_size + 2).astype(np.float32)[1:-1]
    window_2d = np.maximum(np.outer(window_1d, window_1d), 1e-3)[..., None]
    autocast_enabled = amp and device.type == "cuda"
    for top in y_starts:
        for left in x_starts:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = image_u8[top:bottom, left:right]
            tile_h, tile_w = tile.shape[:2]
            tensor = torch.from_numpy(np.ascontiguousarray(tile)).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
            pad_h = tile_size - tile_h
            pad_w = tile_size - tile_w
            if pad_h or pad_w:
                tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                prediction = model(tensor).clamp_(0.0, 1.0)
            prediction = prediction[0, :, :tile_h, :tile_w].permute(1, 2, 0).float().cpu().numpy()
            weight = window_2d[:tile_h, :tile_w]
            output[top:bottom, left:right] += prediction * weight
            weight_sum[top:bottom, left:right] += weight
    return np.clip(output / np.maximum(weight_sum, 1e-8), 0.0, 1.0)


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def save_sidd_comparison(
    noisy_u8: np.ndarray,
    denoised: np.ndarray,
    gt_u8: np.ndarray,
    path: str | Path,
    zoom_size: int = 256,
    panel_width: int = 640,
) -> None:
    """保存 Noisy/Output/GT/Error 全图缩略图与同位置局部放大。"""

    noisy = np.asarray(noisy_u8, dtype=np.float32) / 255.0
    gt = np.asarray(gt_u8, dtype=np.float32) / 255.0
    denoised = np.asarray(denoised, dtype=np.float32)
    error = np.clip(np.abs(denoised - gt) * 4.0, 0.0, 1.0)
    arrays = [noisy, denoised, gt, error]
    labels = ["Noisy", "Output", "GT", "|Output-GT| x4"]
    height, width = noisy.shape[:2]
    zoom = min(zoom_size, height, width)
    top = max(0, (height - zoom) // 2)
    left = max(0, (width - zoom) // 2)
    panel_height = max(1, round(height * panel_width / width))

    canvas = Image.new("RGB", (panel_width * 4, panel_height + panel_width + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate(zip(arrays, labels)):
        image = Image.fromarray(np.clip(np.rint(array * 255), 0, 255).astype(np.uint8), mode="RGB")
        overview = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        crop = image.crop((left, top, left + zoom, top + zoom)).resize(
            (panel_width, panel_width), Image.Resampling.NEAREST
        )
        x = index * panel_width
        canvas.paste(overview, (x, 24))
        canvas.paste(crop, (x, 24 + panel_height))
        draw.text((x + 8, 5), label, fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def paired_bootstrap_ci(values: list[float], seed: int = 42, samples: int = 10_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)
