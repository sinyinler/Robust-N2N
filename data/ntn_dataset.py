from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.discovery import (
    DEFAULT_DATA_SUBDIRS,
    SUPPORTED_ARRAY_EXTS,
    discover_sequence_dirs,
    list_supported_files,
    load_2d,
    natural_key,
)
from utils.intensity import IntensityTransform, lambda_condition_value


def mix_sources_from_args(args) -> list[dict]:
    """从训练参数构造 extra_sources：把 mix 下指定场景（脑/腿）作为额外数据源。

    用法：--mix_root /mnt2/songyd/mix --mix_scenes 305 306 ... 321
    （只取这些编号，不会把 mix 里其它几百个场景或留作 OOD 的手 325 带进来）。
    """
    mix_root = getattr(args, "mix_root", "") or ""
    if not mix_root:
        return []
    subdirs = getattr(args, "mix_subdirs", None)
    if not subdirs:
        subdirs = getattr(args, "data_subdirs", None)
    if not subdirs:
        one = getattr(args, "data_subdir", None)  # train_n2n 用单数 --data_subdir
        subdirs = [one] if one else ["npy", "lbf"]
    return [{
        "root": mix_root,
        "data_subdirs": tuple(subdirs),
        "scenes": getattr(args, "mix_scenes", None),
        "strict_data_subdir": False,
    }]


def center_crop(img: np.ndarray, crop_size: int | None) -> np.ndarray:
    if crop_size is None or crop_size <= 0:
        return img
    h, w = img.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Image size {(h, w)} is smaller than crop_size={crop_size}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return img[top:top + crop_size, left:left + crop_size]


def random_crop_same(images: list[np.ndarray], crop_size: int | None) -> list[np.ndarray]:
    if crop_size is None or crop_size <= 0:
        return images
    h, w = images[0].shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Image size {(h, w)} is smaller than crop_size={crop_size}")
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return [img[top:top + crop_size, left:left + crop_size] for img in images]


def augment_same(images: list[torch.Tensor]) -> list[torch.Tensor]:
    """对 I1/I2/Ĉ 使用相同几何增强，避免错位。"""

    if random.random() < 0.5:
        images = [torch.flip(x, dims=(-1,)) for x in images]
    if random.random() < 0.5:
        images = [torch.flip(x, dims=(-2,)) for x in images]
    k = random.randint(0, 3)
    if k:
        images = [torch.rot90(x, k=k, dims=(-2, -1)) for x in images]
    return images


@dataclass(frozen=True)
class SequenceRecord:
    folder: Path
    files: tuple[Path, ...]


class N2NBootstrapTripletDataset(Dataset):
    """返回 NTN 训练需要的三元组：I1、I2、Ĉ。

    I1/I2 是同一场景的两路独立 noisy observation；Ĉ 是无 GT 条件下的伪干净图。
    默认 Ĉ 用多帧均值获得，也可以在训练脚本里用已训练 N2N 模型输出替换。
    """

    def __init__(
        self,
        root_dir: str,
        intervals: list[int] | tuple[int, ...] = (5, 7, 9),
        crop_size: int = 128,
        random_crop: bool = True,
        pseudo_clean_frames: int = 0,
        data_subdirs: list[str] | tuple[str, ...] = DEFAULT_DATA_SUBDIRS,
        strict_data_subdir: bool = False,
        data_index_min: int | None = None,
        data_index_max: int | None = None,
        include_levels: tuple[int, ...] | None = None,
        extra_sources: list[dict] | None = None,
        intensity_transform: str = "log1p",
        boxcox_lam: float = -0.15,
        boxcox_eps: float = 1e-6,
        lambda_conditioned: bool = False,
        lambda_min: float = -0.3,
        lambda_max: float = 0.2,
        lambda_candidates: list[float] | tuple[float, ...] | None = None,
        vst_lut: str = "",
        augment: bool = True,
        compute_pseudo_clean: bool = True,
    ):
        self.root_dir = root_dir
        self.intervals = [int(x) for x in intervals if int(x) > 0]
        if not self.intervals:
            raise ValueError("At least one positive interval is required")
        self.crop_size = int(crop_size)
        self.random_crop = bool(random_crop)
        self.pseudo_clean_frames = int(pseudo_clean_frames)
        # 训练时若用 N2N(I1) 当 Ĉ（--bootstrap_checkpoint），数据集这里算的多帧均值会被覆盖丢弃。
        # 长序列下「每个样本读全序列帧求均值」是巨大的无用 I/O，故此时关掉，Ĉ 用 I1 占位。
        self.compute_pseudo_clean = bool(compute_pseudo_clean)
        self.lambda_conditioned = bool(lambda_conditioned) and intensity_transform == "boxcox"
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.lambda_candidates = tuple(float(x) for x in lambda_candidates) if lambda_candidates else None
        self.augment = bool(augment)
        self.transform = IntensityTransform(
            name=intensity_transform,
            boxcox_lam=boxcox_lam,
            boxcox_eps=boxcox_eps,
            vst_lut=vst_lut,
        )

        # 主数据源（如 5x5，可带 include_levels）；extra_sources 追加更多根目录
        # （如 mix 下指定的脑/腿场景），实现「多被试混合训练」。
        folders = discover_sequence_dirs(
            root=root_dir,
            data_subdirs=tuple(data_subdirs),
            strict_data_subdir=strict_data_subdir,
            data_index_min=data_index_min,
            data_index_max=data_index_max,
            include_levels=tuple(int(x) for x in include_levels) if include_levels else None,
        )
        for src in (extra_sources or []):
            folders = folders + discover_sequence_dirs(
                root=src["root"],
                data_subdirs=tuple(src.get("data_subdirs", data_subdirs)),
                strict_data_subdir=bool(src.get("strict_data_subdir", strict_data_subdir)),
                include_levels=tuple(int(x) for x in src["levels"]) if src.get("levels") else None,
                include_scenes=tuple(str(x) for x in src["scenes"]) if src.get("scenes") else None,
            )
        self.records: list[SequenceRecord] = []
        self.items: list[tuple[int, int]] = []
        for folder in folders:
            files = tuple(list_supported_files(folder))
            if len(files) <= min(self.intervals):
                continue
            record_idx = len(self.records)
            self.records.append(SequenceRecord(folder=folder, files=files))
            for frame_idx in range(len(files)):
                self.items.append((record_idx, frame_idx))

        if not self.items:
            raise RuntimeError(f"No usable frame sequences found under {root_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def _pair_index(self, frame_idx: int, interval: int, length: int) -> int:
        if frame_idx + interval < length:
            return frame_idx + interval
        if frame_idx - interval >= 0:
            return frame_idx - interval
        return frame_idx

    def _pseudo_indices(self, center_idx: int, length: int) -> list[int]:
        if self.pseudo_clean_frames <= 0 or self.pseudo_clean_frames >= length:
            return list(range(length))
        half = self.pseudo_clean_frames // 2
        start = max(0, center_idx - half)
        end = min(length, start + self.pseudo_clean_frames)
        start = max(0, end - self.pseudo_clean_frames)
        return list(range(start, end))

    def _sample_lambda(self) -> float:
        if not self.lambda_conditioned:
            return self.transform.boxcox_lam
        if self.lambda_candidates is not None:
            return random.choice(self.lambda_candidates)
        return random.uniform(self.lambda_min, self.lambda_max)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record_idx, frame_idx = self.items[idx]
        record = self.records[record_idx]
        files = record.files
        interval = random.choice(self.intervals)
        target_idx = self._pair_index(frame_idx, interval, len(files))

        i1 = load_2d(files[frame_idx])
        i2 = load_2d(files[target_idx])
        if self.compute_pseudo_clean:
            pseudo_stack = [load_2d(files[j]) for j in self._pseudo_indices(frame_idx, len(files))]
            chat = np.mean(np.stack(pseudo_stack, axis=0), axis=0).astype(np.float32, copy=False)
        else:
            # 占位：训练脚本会用 N2N(I1) 覆盖它；这里不再读全序列，避免无用 I/O。
            chat = i1.copy()

        if self.random_crop:
            i1, i2, chat = random_crop_same([i1, i2, chat], self.crop_size)
        else:
            i1, i2, chat = [center_crop(x, self.crop_size) for x in (i1, i2, chat)]

        tensors = [torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0) for x in (i1, i2, chat)]
        if self.augment:
            tensors = augment_same(tensors)

        lam = self._sample_lambda()
        i1_t, i2_t, chat_t = [self.transform.forward(x, lam=lam) for x in tensors]
        condition = None
        if self.lambda_conditioned:
            value = lambda_condition_value(lam, self.lambda_min, self.lambda_max)
            condition = torch.full_like(i1_t, fill_value=value)

        return {
            "input": i1_t,
            "target": i2_t,
            "pseudo_clean": chat_t,
            "condition": condition if condition is not None else torch.empty(0),
            "folder": str(record.folder),
        }
