from __future__ import annotations

"""OOD 泛化评估：在留出的最噪层级（默认 level1）上对比 N2N 基线 vs NTN(T->D')。

实验逻辑（见 README「泛化对照实验协议」）：
- N2N 与 NTN 都只在 level 2/3/4 上训练，level1 对两者都是 OOD（从未见过的更强噪声）。
- 同一物理场景在各层级共享（用户确认），所以可用**同场景最高叠加层(默认 level4)的多帧均值**
  作为伪干净 GT，对 level1 的去噪结果算 PSNR/SSIM。
- 公平起见，两种方法都在 log1p 域、对同一个伪 GT 计算指标。

输出：
- results/eval_ood/metrics_level{L}.json：N2N vs NTN 的平均 PSNR/SSIM 及逐场景明细。
- results/eval_ood/compare/scene{idx}_frame{f}.png：noisy / N2N / NTN / GT 四联 + 血管局部放大。
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.discovery import discover_sequence_dirs, list_supported_files, load_2d
from models.denoiser import Denoiser
from models.ntn import NoiseTranslator
from utils.checkpoint import load_weights_flexible
from utils.metrics import center_crop_to_match, psnr, ssim_simple

LEVEL_DIR_RE = re.compile(r"\d+x\d+x(\d+)$")

# 优先用 skimage 的窗口化 SSIM（更可信）；没装则退回项目自带的全图近似 SSIM。
try:
    from skimage.metrics import structural_similarity as _sk_ssim

    def ssim(a, b, data_range):
        a, b = center_crop_to_match(np.asarray(a), np.asarray(b))
        return float(_sk_ssim(a.astype(np.float64), b.astype(np.float64), data_range=data_range))

    SSIM_BACKEND = "skimage"
except Exception:
    def ssim(a, b, data_range):
        return float(ssim_simple(a, b, data_range=data_range))

    SSIM_BACKEND = "simple"


def log1p_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32, copy=False)


def parse_level_scene(folder: Path) -> tuple[int, str] | None:
    """从 .../5x5x{N}/{scene}/npy 这样的路径里解析 (level=N, scene)。"""

    parts = folder.parts
    for i, p in enumerate(parts):
        m = LEVEL_DIR_RE.match(p)
        if m and i + 1 < len(parts):
            return int(m.group(1)), parts[i + 1]
    return None


def build_scene_index(data_path: str, data_subdirs, strict: bool) -> dict[str, dict[int, Path]]:
    """返回 {scene: {level: folder}}，便于按场景对齐不同层级。"""

    folders = discover_sequence_dirs(
        root=data_path, data_subdirs=tuple(data_subdirs), strict_data_subdir=strict
    )
    scene_map: dict[str, dict[int, Path]] = defaultdict(dict)
    for folder in folders:
        ls = parse_level_scene(folder)
        if ls is None:
            continue
        level, scene = ls
        scene_map[scene][level] = folder
    return scene_map


@torch.no_grad()
def make_pseudo_gt(folder: Path, max_frames: int) -> np.ndarray:
    """伪干净 GT：高叠加层多帧在 raw 域取均值（线性域平均才能正确压噪），再交给调用方转 log1p。"""

    files = list_supported_files(folder)
    if max_frames > 0:
        files = files[:max_frames]
    stack = [load_2d(f) for f in files]
    return np.mean(np.stack(stack, axis=0), axis=0).astype(np.float32)


@torch.no_grad()
def denoise_n2n(model, z: np.ndarray, device) -> np.ndarray:
    t = torch.from_numpy(z).float().unsqueeze(0).unsqueeze(0).to(device)
    return model(t).squeeze(0).squeeze(0).cpu().numpy()


@torch.no_grad()
def denoise_ntn(translator, expert, z: np.ndarray, device) -> np.ndarray:
    t = torch.from_numpy(z).float().unsqueeze(0).unsqueeze(0).to(device)
    translated = translator(t)
    return expert(translated).squeeze(0).squeeze(0).cpu().numpy()


def _norm_u8(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.uint8)
    y = np.clip((x.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def save_panels(named_images: list[tuple[str, np.ndarray]], out_path: Path, zoom_size: int) -> None:
    """多联对比图（上排全图 + 下排中心放大），所有图用同一窗宽窗位，对比公平。"""

    arrs = [img for _, img in named_images]
    base = arrs[0]
    arrs = [center_crop_to_match(a, base)[0] for a in arrs]
    cat = np.concatenate([a.reshape(-1) for a in arrs])
    vmin, vmax = float(np.percentile(cat, 1)), float(np.percentile(cat, 99))

    h, w = arrs[0].shape
    zoom = min(zoom_size, h, w)
    top, left = max(0, (h - zoom) // 2), max(0, (w - zoom) // 2)

    panels, zooms = [], []
    for a in arrs:
        p = Image.fromarray(_norm_u8(a, vmin, vmax), mode="L").convert("RGB")
        d = ImageDraw.Draw(p)
        d.rectangle([left, top, left + zoom, top + zoom], outline=(255, 0, 0), width=max(1, w // 256))
        z = p.crop((left, top, left + zoom, top + zoom)).resize((w, h), Image.Resampling.BICUBIC)
        panels.append(p)
        zooms.append(z)

    from PIL import ImageFont
    canvas = Image.new("RGB", (w * len(panels), h * 2 + 24), color=(0, 0, 0))
    for i, (name, _) in enumerate(named_images):
        canvas.paste(panels[i], (i * w, 24))
        canvas.paste(zooms[i], (i * w, h + 24))
        ImageDraw.Draw(canvas).text((i * w + 5, 5), name, fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main(args) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    n2n = Denoiser(input_channels=1).to(device).eval()
    print("[INFO] N2N:", load_weights_flexible(n2n, args.n2n_checkpoint, device))
    translator = NoiseTranslator(
        input_channels=1, width=args.width, middle_blocks=args.middle_blocks,
        inject_sigma=args.inject_sigma, residual_scale=args.residual_scale,
    ).to(device).eval()
    print("[INFO] T:", load_weights_flexible(translator, args.translator_checkpoint, device))
    expert = Denoiser(input_channels=1).to(device).eval()
    print("[INFO] D':", load_weights_flexible(expert, args.gaussian_expert_checkpoint, device))

    scene_map = build_scene_index(args.data_path, args.data_subdirs, bool(args.strict_data_subdir))
    all_levels = sorted({lv for d in scene_map.values() for lv in d})
    gt_level = args.gt_level if args.gt_level > 0 else max(all_levels)

    # 可选：用一张固定的 reference.npy 当 GT（比同场景 level4 多帧均值更干净）。
    # 它必须与被评估场景像素对齐，因此通常配合 --scenes 限定到对应场景（如 --scenes 0）。
    fixed_gt_z = None
    if args.reference_npy:
        fixed_gt_z = log1p_np(load_2d(Path(args.reference_npy)))
        print(f"[INFO] 使用固定参考 GT: {args.reference_npy}  (忽略 gt_level 多帧均值)")
    scene_filter = set(args.scenes) if args.scenes else None
    print(f"[INFO] levels present={all_levels}; eval_level={args.eval_level}, "
          f"gt={'reference.npy' if fixed_gt_z is not None else 'level%d-mean'%gt_level}, SSIM={SSIM_BACKEND}")

    rows = {"noisy": [], "n2n": [], "ntn": []}  # 每项: (psnr, ssim)
    per_scene = []
    vis_done = 0
    compare_dir = Path(args.out_dir) / "compare"

    scenes = sorted(scene_map.keys(), key=lambda s: int(s) if s.isdigit() else s)
    for scene in scenes:
        levels = scene_map[scene]
        if scene_filter is not None and scene not in scene_filter:
            continue
        if args.eval_level not in levels:
            continue
        if fixed_gt_z is None and gt_level not in levels:
            continue
        if args.max_scenes > 0 and len(per_scene) >= args.max_scenes:
            break

        if fixed_gt_z is not None:
            gt_z = fixed_gt_z
        else:
            gt_z = log1p_np(make_pseudo_gt(levels[gt_level], args.gt_frames))
        dr = float(gt_z.max() - gt_z.min()) or 1.0  # log1p 域、用 GT 的动态范围作 data_range

        files = list_supported_files(levels[args.eval_level])
        if args.max_frames_per_scene > 0:
            files = files[:args.max_frames_per_scene]

        s_acc = {"noisy": [], "n2n": [], "ntn": []}
        for fi, f in enumerate(files):
            noisy_z = log1p_np(load_2d(f))
            n2n_z = denoise_n2n(n2n, noisy_z, device)
            ntn_z = denoise_ntn(translator, expert, noisy_z, device)

            for key, img in (("noisy", noisy_z), ("n2n", n2n_z), ("ntn", ntn_z)):
                s_acc[key].append((psnr(img, gt_z, data_range=dr), ssim(img, gt_z, data_range=dr)))

            if vis_done < args.max_vis_scenes and fi == 0:
                gt_label = "GT(reference)" if fixed_gt_z is not None else "pseudo-GT(level%d)" % gt_level
                save_panels(
                    [("noisy(level%d)" % args.eval_level, noisy_z), ("N2N", n2n_z),
                     ("NTN(ours)", ntn_z), (gt_label, gt_z)],
                    compare_dir / f"scene{scene}_frame0.png", args.zoom_size,
                )

        scene_means = {k: np.mean(v, axis=0) for k, v in s_acc.items()}  # 每 key: [psnr, ssim]
        for k in rows:
            rows[k].append(scene_means[k])
        per_scene.append({"scene": scene, **{k: scene_means[k].tolist() for k in scene_means}})
        if vis_done < args.max_vis_scenes:
            vis_done += 1

    if not per_scene:
        raise RuntimeError(f"没有可评估的场景（eval_level={args.eval_level}, scenes={args.scenes}），检查数据/层级/筛选。")

    gt_desc = "reference.npy" if fixed_gt_z is not None else f"pseudo-GT level{gt_level}"
    summary = {k: np.mean(rows[k], axis=0).tolist() for k in rows}
    print(f"\n==== OOD eval on level{args.eval_level} (vs {gt_desc}, {len(per_scene)} scenes) ====")
    print(f"{'method':>14} | {'PSNR(dB)':>9} | {'SSIM':>7}")
    print("-" * 36)
    label = {"noisy": "noisy input", "n2n": "N2N (lv234)", "ntn": "NTN (ours)"}
    for k in ("noisy", "n2n", "ntn"):
        print(f"{label[k]:>14} | {summary[k][0]:>9.3f} | {summary[k][1]:>7.4f}")
    gain_psnr = summary["ntn"][0] - summary["n2n"][0]
    gain_ssim = summary["ntn"][1] - summary["n2n"][1]
    print("-" * 36)
    print(f"{'NTN - N2N':>14} | {gain_psnr:>+9.3f} | {gain_ssim:>+7.4f}   <- OOD 泛化增益")

    out = {
        "eval_level": args.eval_level, "gt_level": gt_level, "ssim_backend": SSIM_BACKEND,
        "n_scenes": len(per_scene),
        "summary": {label[k]: {"psnr": summary[k][0], "ssim": summary[k][1]} for k in rows},
        "gain_ntn_minus_n2n": {"psnr": gain_psnr, "ssim": gain_ssim},
        "per_scene": per_scene,
    }
    out_path = Path(args.out_dir) / f"metrics_level{args.eval_level}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] 指标写入 {out_path}；对比图在 {compare_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOD generalization eval: N2N baseline vs NTN on held-out level.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--data_subdirs", nargs="*", default=["npy", "lbf"])
    p.add_argument("--strict_data_subdir", type=int, default=1)
    p.add_argument("--eval_level", type=int, default=1, help="留出的 OOD 评估层级（默认最噪 level1）。")
    p.add_argument("--gt_level", type=int, default=0, help="伪 GT 用的最高叠加层级；<=0 自动取最大。")
    p.add_argument("--gt_frames", type=int, default=0, help="伪 GT 多帧均值用多少帧；<=0 用全部。")
    p.add_argument("--reference_npy", type=str, default="",
                   help="用一张固定 reference.npy 当 GT（替代同场景多帧均值）；需与场景像素对齐，常配合 --scenes。")
    p.add_argument("--scenes", type=str, nargs="*", default=None,
                   help="只评估这些场景编号（如 --scenes 0）。配合 --reference_npy 用。")
    p.add_argument("--max_frames_per_scene", type=int, default=3, help="每个场景评估多少帧 OOD 输入。")
    p.add_argument("--max_scenes", type=int, default=0, help="最多评估多少场景；<=0 不限。")
    p.add_argument("--max_vis_scenes", type=int, default=12, help="出多少张对比图。")
    p.add_argument("--zoom_size", type=int, default=128)
    p.add_argument("--n2n_checkpoint", type=str, required=True)
    p.add_argument("--translator_checkpoint", type=str, required=True)
    p.add_argument("--gaussian_expert_checkpoint", type=str, required=True)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--middle_blocks", type=int, default=2)
    p.add_argument("--inject_sigma", type=float, default=1.0)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--out_dir", type=str, default="results/eval_ood")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
