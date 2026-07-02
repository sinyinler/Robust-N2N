from __future__ import annotations

"""测量真实 BFI 数据的噪声水平，用于给 Gaussian expert D' 选一个对的固定 sigma。

为什么这么测（务必先读）：
---------------------------------------------------------------------------
NTN 框架里，去噪器 D' 被训练成「只会处理某个固定 sigma 的高斯噪声」的偏科生，
翻译器 T 负责把真实噪声搬到这个工作点。所以这个 sigma 必须等于「真实噪声在
模型实际看到的域里的强度」，否则 D' 训练时见到的噪声和推理时 T 吐出来的噪声
对不上，泛化就无从谈起。

模型实际看到的域 = log1p 域（和 train_gaussian_expert / train_translator 默认一致）。
因此本脚本默认在 log1p 域估计噪声，而不是在 raw 域。

怎么定义「噪声」：同场景相邻两帧做差。
    f_k = S + n_k,  f_{k+1} = S + n_{k+1}   （同场景，静态结构 S 帧间一致）
    d = f_{k+1} - f_k = n_{k+1} - n_k,       Var(d) = 2 * sigma^2
    => sigma = std(d) / sqrt(2)
相邻帧（间隔最小）能把血流/运动带来的「信号漂移」压到最低；而帧间随机起伏本来
就是 N2N 当作噪声去掉、NTN 要 Gaussian 化的那部分，所以这个估计正是 D' 的目标
工作点。静态血管在做差里相消、不会被算进噪声，符合「不能磨掉细小血管」的诉求。

稳健性：除了普通 std，还用 MAD（中位数绝对偏差）估计，
    sigma_mad = median(|d - median(d)|) / 0.6745 / sqrt(2)
对运动/异常像素更不敏感，推荐以 MAD 结果为准。

按叠加层级分组：目录名形如 5x5x{N} 时，N 通常是叠加帧数，噪声应随 N 增大而下降
（~1/sqrt(N)）。分组输出能看清这个规律，并据此决定是用单一 sigma 还是按层级设窄带。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 复用项目自带的「数据发现 + 读取」逻辑（纯 numpy，无需 torch），保证和训练时对数据的理解一致。
from data.discovery import discover_sequence_dirs, list_supported_files, load_2d  # noqa: E402

MAD_TO_SIGMA = 0.6744897501960817  # 标准正态的 0.75 分位数，MAD/此值 ≈ std
LEVEL_RE = re.compile(r"(\d+x\d+x(\d+))")


def to_domain(arr: np.ndarray, transform: str) -> np.ndarray:
    """把 raw 数据变到模型实际看到的域。默认 log1p，与训练默认一致。"""

    if transform == "none":
        return arr.astype(np.float32, copy=False)
    if transform == "log1p":
        return np.log1p(np.clip(arr, 0.0, None)).astype(np.float32, copy=False)
    raise ValueError(f"unsupported intensity_transform for this script: {transform}")


def infer_level(folder: Path) -> str:
    """从路径里推断叠加层级标签（如 '5x5x100' 里的 100）。推不出就归到 'flat'。"""

    best = None
    for part in folder.parts:
        m = LEVEL_RE.search(part)
        if m:
            best = int(m.group(2))  # 取最里层匹配，5x5xN 的 N
    return str(best) if best is not None else "flat"


def center_crop(img: np.ndarray, crop: int) -> np.ndarray:
    if crop <= 0:
        return img
    h, w = img.shape
    if h <= crop and w <= crop:
        return img
    top = max(0, (h - crop) // 2)
    left = max(0, (w - crop) // 2)
    return img[top:top + crop, left:left + crop]


def sequence_sigma(files, transform: str, crop: int, max_frames: int) -> dict | None:
    """对一条同场景序列，用相邻帧做差估计噪声 sigma。"""

    n = len(files) if max_frames <= 0 else min(len(files), max_frames)
    if n < 2:
        return None

    frames = []
    for path in files[:n]:
        try:
            arr = load_2d(path)
        except Exception as exc:  # 单帧坏掉不应让整次测量失败
            print(f"[WARN] skip unreadable frame {path}: {exc}")
            continue
        frames.append(center_crop(to_domain(arr, transform), crop))
    if len(frames) < 2:
        return None

    # 收集所有相邻帧差，统一估计（更稳）。
    diffs = []
    for k in range(len(frames) - 1):
        if frames[k].shape != frames[k + 1].shape:
            continue
        diffs.append((frames[k + 1] - frames[k]).reshape(-1))
    if not diffs:
        return None
    d = np.concatenate(diffs).astype(np.float32)

    sigma_std = float(np.std(d) / np.sqrt(2.0))
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma_mad = float(mad / MAD_TO_SIGMA / np.sqrt(2.0))
    signal_std = float(np.std(frames[0]))  # 该域下信号本身的尺度，便于换算相对噪声

    return {
        "n_frames": len(frames),
        "n_pairs": len(diffs),
        "sigma_std": sigma_std,
        "sigma_mad": sigma_mad,
        "signal_std": signal_std,
    }


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def main(args) -> None:
    folders = discover_sequence_dirs(
        root=args.data_path,
        data_subdirs=tuple(args.data_subdirs),
        strict_data_subdir=bool(args.strict_data_subdir),
        data_index_min=args.data_index_min if args.data_index_min >= 0 else None,
        data_index_max=args.data_index_max if args.data_index_max >= 0 else None,
    )
    if not folders:
        raise RuntimeError(f"No sequence folders found under {args.data_path}")

    # 可选：只测指定场景（按场景文件夹名筛选，如 mix 下的 305 306 ... 这种 flat 多帧结构）。
    subdir_set = {s.lower() for s in args.data_subdirs}
    include_scenes = set(str(s) for s in args.include_scenes) if args.include_scenes else None

    def scene_name(folder: Path) -> str:
        # mix/305/npy -> "305"；若 leaf 本身是场景目录则取其名。
        return folder.parent.name if folder.name.lower() in subdir_set else folder.name

    # 按层级分组并对每组限制序列数量，避免在大数据集上跑太久。
    per_level_count: dict[str, int] = {}
    records: list[dict] = []
    if include_scenes is not None:
        matched = sum(1 for f in folders if scene_name(f) in include_scenes)
        print(f"[INFO] discovered {len(folders)} total; {matched} match --include_scenes {sorted(include_scenes)}")
    else:
        print(f"[INFO] discovered {len(folders)} candidate sequence folders")
    for folder in folders:
        if include_scenes is not None and scene_name(folder) not in include_scenes:
            continue
        level = infer_level(folder)
        if args.max_seqs_per_level > 0 and per_level_count.get(level, 0) >= args.max_seqs_per_level:
            continue
        files = list_supported_files(folder)
        stat = sequence_sigma(files, args.intensity_transform, args.crop, args.max_frames_per_seq)
        if stat is None:
            continue
        stat["folder"] = str(folder)
        stat["level"] = level
        records.append(stat)
        per_level_count[level] = per_level_count.get(level, 0) + 1

    if not records:
        raise RuntimeError("No usable sequences (need >=2 frames each).")

    # 分层汇总（按层级数值排序，flat 放最后）。
    levels = sorted({r["level"] for r in records}, key=lambda s: (s == "flat", int(s) if s.isdigit() else 0))
    per_level_summary: dict[str, dict] = {}
    print("\n================ Noise level by acquisition level (domain="
          f"{args.intensity_transform}) ================")
    header = f"{'level':>8} | {'#seq':>5} | {'sigma_mad(med)':>14} | {'sigma_std(med)':>14} | {'p25..p75 (mad)':>20} | {'signal_std(med)':>15}"
    print(header)
    print("-" * len(header))
    for level in levels:
        rs = [r for r in records if r["level"] == level]
        mad_summ = summarize([r["sigma_mad"] for r in rs])
        std_summ = summarize([r["sigma_std"] for r in rs])
        sig_summ = summarize([r["signal_std"] for r in rs])
        per_level_summary[level] = {"sigma_mad": mad_summ, "sigma_std": std_summ, "signal_std": sig_summ}
        print(f"{level:>8} | {mad_summ['count']:>5} | {mad_summ['median']:>14.5f} | "
              f"{std_summ['median']:>14.5f} | {mad_summ['p25']:>9.5f}..{mad_summ['p75']:<9.5f} | "
              f"{sig_summ['median']:>15.4f}")

    overall_mad = summarize([r["sigma_mad"] for r in records])
    overall_std = summarize([r["sigma_std"] for r in records])
    print("-" * len(header))
    print(f"{'ALL':>8} | {overall_mad['count']:>5} | {overall_mad['median']:>14.5f} | "
          f"{overall_std['median']:>14.5f} | {overall_mad['p25']:>9.5f}..{overall_mad['p75']:<9.5f} |")

    print("\n[建议] 单一固定 sigma 取全体 sigma_mad 中位数: "
          f"{overall_mad['median']:.4f}  (窄带可取 {overall_mad['p25']:.4f} ~ {overall_mad['p75']:.4f})")
    print("[提示] 若各层级 sigma 差异很大，建议按层级设窄带、或训练时按层级采样，而不是单一值。")

    out = {
        "data_path": str(args.data_path),
        "intensity_transform": args.intensity_transform,
        "crop": args.crop,
        "max_frames_per_seq": args.max_frames_per_seq,
        "max_seqs_per_level": args.max_seqs_per_level,
        "overall": {"sigma_mad": overall_mad, "sigma_std": overall_std},
        "per_level": per_level_summary,
        "recommended_sigma": overall_mad["median"],
        "recommended_band": [overall_mad["p25"], overall_mad["p75"]],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] 详细统计已写入 {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure real BFI noise level (log1p domain) to set D' sigma.")
    p.add_argument("--data_path", type=str, required=True, help="如 /mnt2/songyd/5x5（层级/场景/npy 结构）。")
    p.add_argument("--data_subdirs", nargs="*", default=["npy", "lbf"])
    p.add_argument("--strict_data_subdir", type=int, default=0)
    p.add_argument("--data_index_min", type=int, default=-1)
    p.add_argument("--data_index_max", type=int, default=-1)
    p.add_argument("--include_scenes", nargs="*", default=None,
                   help="只测这些场景文件夹名（如 mix 下 --include_scenes 305 306 ... 325）。")
    p.add_argument("--intensity_transform", choices=["none", "log1p"], default="log1p",
                   help="默认 log1p，与训练保持一致；用 none 看 raw 域噪声。")
    p.add_argument("--crop", type=int, default=512, help="中心裁剪边长，加速并避开边缘；<=0 用整图。")
    p.add_argument("--max_frames_per_seq", type=int, default=12, help="每条序列最多用多少帧；<=0 用全部。")
    p.add_argument("--max_seqs_per_level", type=int, default=40, help="每个叠加层级最多采样多少条序列；<=0 不限。")
    p.add_argument("--out", type=str, default="results/eval/noise_stats.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
