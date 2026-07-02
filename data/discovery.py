from __future__ import annotations

"""数据发现与读取的纯 numpy helper（不依赖 torch）。

把这些和具体张量/增强无关的逻辑单独抽出来，既能被 N2NBootstrapTripletDataset 复用，
也能被 scripts/measure_noise.py 这类离线分析脚本在不安装 torch 的环境下直接调用。
"""

import re
from pathlib import Path

import numpy as np

from utils.lbfreadnew import lbfreadnew


SUPPORTED_ARRAY_EXTS = (".npy", ".lbf")
DEFAULT_DATA_SUBDIRS = ("npy", "lbf")


def natural_key(path: Path):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)]


def list_supported_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_ARRAY_EXTS],
        key=natural_key,
    )


def load_2d(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".lbf":
        arr = lbfreadnew(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path}")

    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{path} shape {arr.shape} is not 2D after squeeze")
    return arr.astype(np.float32, copy=False)


def parse_level_name(name: str) -> int | None:
    match = re.match(r"^\d+x\d+x(\d+)$", name)
    return int(match.group(1)) if match else None


def parse_index_name(name: str) -> int | None:
    return int(name) if re.fullmatch(r"\d+", name) else None


def index_allowed(name: str, index_min: int | None, index_max: int | None) -> bool:
    if index_min is None and index_max is None:
        return True
    index = parse_index_name(name)
    if index is None:
        return False
    if index_min is not None and index < index_min:
        return False
    if index_max is not None and index > index_max:
        return False
    return True


def scene_name_of(folder: Path, data_subdirs: tuple[str, ...] = DEFAULT_DATA_SUBDIRS) -> str:
    """从序列目录推断「场景编号」：mix/325/npy -> '325'；mix/316 -> '316'。"""
    return folder.parent.name if folder.name.lower() in {s.lower() for s in data_subdirs} else folder.name


def discover_sequence_dirs(
    root: str | Path,
    data_subdirs: tuple[str, ...] = DEFAULT_DATA_SUBDIRS,
    strict_data_subdir: bool = False,
    data_index_min: int | None = None,
    data_index_max: int | None = None,
    include_levels: tuple[int, ...] | None = None,
    include_scenes: tuple[str, ...] | None = None,
) -> list[Path]:
    """发现可形成 N2N pair 的具体帧序列目录。

    兼容两类数据：
    1. root 本身就是一个帧序列目录；
    2. 原项目的 mix/5x5x100/0/npy 这类层级结构。

    include_levels 不为空时，只保留叠加层级 5x5x{N} 的 N 落在该集合里的序列。
    include_scenes 不为空时，只保留场景编号在该集合里的序列（如 mix 下 305..312、316..321）。
    """

    scenes_set = {str(s) for s in include_scenes} if include_scenes else None

    def _filter(dirs: list[Path]) -> list[Path]:
        if scenes_set is None:
            return dirs
        return [d for d in dirs if scene_name_of(d, data_subdirs) in scenes_set]

    root = Path(root)
    if list_supported_files(root):
        if include_levels is not None:
            raise ValueError(
                f"--levels was given but {root} is a single flat sequence with no 5x5xN level info."
            )
        return _filter([root])

    levels_set = set(include_levels) if include_levels is not None else None
    sequence_dirs: list[Path] = []
    for level_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=natural_key):
        level = parse_level_name(level_dir.name)
        if level is None:
            # flat 结构：level_dir 本身就是一个场景目录（如 mix/325 或 mix/316）。
            # 每个编号=一个同场景，帧要么在唯一的数据子目录里（mix/325/npy/*.npy），
            # 要么直接在该目录下（mix/316/*.lbf）。优先用子目录、否则用直接文件，二者其一。
            if levels_set is not None:
                continue
            if not index_allowed(level_dir.name, data_index_min, data_index_max):
                continue
            sub_found = False
            for subdir_name in data_subdirs:
                candidate = level_dir / subdir_name
                if list_supported_files(candidate):
                    sequence_dirs.append(candidate)
                    sub_found = True
            if not sub_found and list_supported_files(level_dir):
                sequence_dirs.append(level_dir)
            continue

        if levels_set is not None and level not in levels_set:
            continue

        for scene_dir in sorted((p for p in level_dir.iterdir() if p.is_dir()), key=natural_key):
            if not index_allowed(scene_dir.name, data_index_min, data_index_max):
                continue
            if not strict_data_subdir and list_supported_files(scene_dir):
                sequence_dirs.append(scene_dir)
            for subdir_name in data_subdirs:
                candidate = scene_dir / subdir_name
                if list_supported_files(candidate):
                    sequence_dirs.append(candidate)

    if sequence_dirs or levels_set is not None:
        # 指定了层级却没找到任何序列时，直接返回空，交给上层报错；
        # 不走 rglob 兜底，以免把被过滤掉的层级又递归捞回来。
        return _filter(sequence_dirs)

    for child in sorted((p for p in root.rglob("*") if p.is_dir()), key=natural_key):
        if list_supported_files(child):
            sequence_dirs.append(child)
    return _filter(sequence_dirs)
