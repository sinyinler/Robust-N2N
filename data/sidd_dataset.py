"""SIDD-Small sRGB 配对发现、scene 隔离和同步裁剪增强。"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SCENE_PATTERN = re.compile(r"^\d{4}_(\d{3})_")


@dataclass(frozen=True)
class SIDDPair:
    name: str
    scene: str
    noisy: Path
    gt: Path


def resolve_sidd_data_dir(root: str | Path) -> Path:
    """兼容用户给外层目录、解压目录或最终 Data 目录三种写法。"""

    root = Path(root).expanduser().resolve()
    direct_candidates = [root, root / "Data", root / "SIDD_Small_sRGB_Only" / "Data"]
    for candidate in direct_candidates:
        if candidate.is_dir() and any(candidate.glob("*/NOISY_SRGB_010.PNG")):
            return candidate
    for candidate in root.rglob("Data"):
        if candidate.is_dir() and any(candidate.glob("*/NOISY_SRGB_010.PNG")):
            return candidate.resolve()
    raise FileNotFoundError(f"在 {root} 下没有找到 SIDD Data/scene/NOISY_SRGB_010.PNG")


def discover_sidd_pairs(root: str | Path, scenes: tuple[str, ...] | list[str] | None = None) -> list[SIDDPair]:
    data_dir = resolve_sidd_data_dir(root)
    allowed = {str(scene).zfill(3) for scene in scenes} if scenes else None
    pairs: list[SIDDPair] = []
    for scene_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        match = SCENE_PATTERN.match(scene_dir.name)
        if match is None:
            continue
        scene = match.group(1)
        if allowed is not None and scene not in allowed:
            continue
        noisy = scene_dir / "NOISY_SRGB_010.PNG"
        gt = scene_dir / "GT_SRGB_010.PNG"
        if not noisy.is_file() or not gt.is_file():
            raise FileNotFoundError(f"SIDD pair 不完整: {scene_dir}")
        with Image.open(noisy) as noisy_image, Image.open(gt) as gt_image:
            if noisy_image.mode != "RGB" or gt_image.mode != "RGB":
                raise ValueError(f"期望 RGB PNG: {scene_dir}")
            if noisy_image.size != gt_image.size:
                raise ValueError(f"NOISY/GT 尺寸不一致: {scene_dir}")
        pairs.append(SIDDPair(scene_dir.name, scene, noisy, gt))
    if not pairs:
        raise ValueError(f"没有发现符合 scenes={scenes} 的 SIDD pair")
    return pairs


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_sidd_pair(pair: SIDDPair) -> tuple[np.ndarray, np.ndarray]:
    noisy = _load_rgb(pair.noisy)
    gt = _load_rgb(pair.gt)
    if noisy.shape != gt.shape:
        raise ValueError(f"运行时检测到 NOISY/GT 尺寸不一致: {pair.name}")
    return noisy, gt


class SIDDSceneCropDataset(Dataset):
    """每次解码一对大图并返回多个 crop，减少重复 PNG 解码开销。"""

    def __init__(
        self,
        root: str | Path,
        scenes: tuple[str, ...] | list[str],
        crop_size: int = 256,
        repeats_per_pair: int = 8,
        crops_per_load: int = 4,
        augment: bool = True,
        deterministic: bool = False,
        seed: int = 42,
    ) -> None:
        self.pairs = discover_sidd_pairs(root, scenes)
        self.crop_size = int(crop_size)
        self.repeats_per_pair = int(repeats_per_pair)
        self.crops_per_load = int(crops_per_load)
        self.augment = bool(augment)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        if self.crop_size <= 0 or self.repeats_per_pair <= 0 or self.crops_per_load <= 0:
            raise ValueError("crop_size/repeats_per_pair/crops_per_load 必须为正数")

    def __len__(self) -> int:
        return len(self.pairs) * self.repeats_per_pair

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index % len(self.pairs)]
        noisy, gt = load_sidd_pair(pair)
        height, width = noisy.shape[:2]
        size = self.crop_size
        if height < size or width < size:
            raise ValueError(f"{pair.name} 尺寸 {(height, width)} 小于 crop_size={size}")

        rng = random.Random(self.seed + index * 1009) if self.deterministic else random
        noisy_crops: list[torch.Tensor] = []
        gt_crops: list[torch.Tensor] = []
        for _ in range(self.crops_per_load):
            top = rng.randint(0, height - size)
            left = rng.randint(0, width - size)
            noisy_crop = noisy[top : top + size, left : left + size]
            gt_crop = gt[top : top + size, left : left + size]

            if self.augment:
                if rng.random() < 0.5:
                    noisy_crop, gt_crop = noisy_crop[:, ::-1], gt_crop[:, ::-1]
                if rng.random() < 0.5:
                    noisy_crop, gt_crop = noisy_crop[::-1], gt_crop[::-1]
                rotations = rng.randrange(4)
                if rotations:
                    noisy_crop = np.rot90(noisy_crop, rotations)
                    gt_crop = np.rot90(gt_crop, rotations)

            noisy_tensor = torch.from_numpy(np.ascontiguousarray(noisy_crop)).permute(2, 0, 1).float().div_(255.0)
            gt_tensor = torch.from_numpy(np.ascontiguousarray(gt_crop)).permute(2, 0, 1).float().div_(255.0)
            noisy_crops.append(noisy_tensor)
            gt_crops.append(gt_tensor)

        return {
            "noisy": torch.stack(noisy_crops),
            "gt": torch.stack(gt_crops),
            "name": pair.name,
            "scene": pair.scene,
        }


def seed_sidd_worker(worker_id: int) -> None:
    """让 Python/NumPy 增强随机数与 DataLoader 为 worker 分配的 seed 对齐。"""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
