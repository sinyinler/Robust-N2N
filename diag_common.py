# -*- coding: utf-8 -*-
"""诊断脚本共用的帧采样。

**关键**：batch 必须由**不同场景**组成，否则
  - batch_shuffle 无效（同场景不同帧本就该有几乎相同的深层特征）；
  - 「跨样本 std」测的是噪声实现间的差异，不是输入间的差异。
默认从 scene_root 下的多个场景各取一帧。
"""
from __future__ import annotations

import re
from pathlib import Path


def natural_key(p: Path):
    m = re.findall(r"\d+", p.stem)
    return (int(m[0]) if m else 0, p.stem)


def collect_frames(scene_root: str | None, n_scenes: int, frame_idx: int,
                   scene_dir: str | None = None, n_frames: int = 0):
    """返回帧路径列表。
    - scene_root 给定：从 scene_root/<scene>/npy 里各取第 frame_idx 帧（**不同场景**，推荐）
    - 否则退回 scene_dir 的前 n_frames 帧（**同场景**，仅供对照，batch_shuffle/跨样本 std 不可解读）
    """
    if scene_root:
        root = Path(scene_root)
        scenes = sorted([d for d in root.iterdir() if d.is_dir()], key=natural_key)[:n_scenes]
        frames = []
        for s in scenes:
            npy_dir = s / "npy" if (s / "npy").is_dir() else s
            fs = sorted(npy_dir.glob("*.npy"), key=natural_key)
            if len(fs) > frame_idx:
                frames.append(fs[frame_idx])
        if len(frames) < 2:
            raise RuntimeError(f"{scene_root} 下只找到 {len(frames)} 个场景，至少需要 2 个")
        print(f"[INFO] 取 {len(frames)} 个**不同场景**各第 {frame_idx} 帧："
              f"{[f.parents[1].name for f in frames]}")
        return frames, True

    frames = sorted(Path(scene_dir).glob("*.npy"), key=natural_key)[:n_frames]
    if len(frames) < 2:
        raise RuntimeError("至少需要 2 帧")
    print(f"[WARN] 使用**同一场景**的 {len(frames)} 帧："
          f"batch_shuffle 与「跨样本 std」在此设置下不可解读（同场景不同帧本就相似）。")
    return frames, False
