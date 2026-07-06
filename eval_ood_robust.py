from __future__ import annotations

"""OOD 泛化评估：在留出的最噪层级（默认 level1）上对比 N2N 基线 vs Robust-N2N。

协议与 eval_ood.py 一致（两者都只在 level 2/3/4 训练，level1 对两者都是 OOD）：
- 伪干净 GT = 同场景最高叠加层(默认最大 level)的多帧均值（raw 域平均后转 log1p）；
- 两方法都在 log1p 域、对同一伪 GT 算 PSNR/SSIM，公平；
- 出 noisy / N2N / Robust-N2N / GT 四联 + 血管中心放大图。

N2N = Denoiser；Robust-N2N = DenoiserWithFeats（无 GIBlock，推理只取去噪输出）。
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
from models.denoiser_feats import DenoiserWithFeats
from utils.checkpoint import load_weights_flexible
from utils.metrics import center_crop_to_match, psnr, ssim_simple

LEVEL_DIR_RE = re.compile(r"\d+x\d+x(\d+)$")

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


def log1p_np(x):
    return np.log1p(np.clip(x, 0.0, None)).astype(np.float32, copy=False)


def parse_level_scene(folder: Path):
    parts = folder.parts
    for i, p in enumerate(parts):
        m = LEVEL_DIR_RE.match(p)
        if m and i + 1 < len(parts):
            return int(m.group(1)), parts[i + 1]
    return None


def build_scene_index(data_path, data_subdirs, strict):
    folders = discover_sequence_dirs(root=data_path, data_subdirs=tuple(data_subdirs), strict_data_subdir=strict)
    scene_map = defaultdict(dict)
    for folder in folders:
        ls = parse_level_scene(folder)
        if ls is None:
            continue
        level, scene = ls
        scene_map[scene][level] = folder
    return scene_map


@torch.no_grad()
def make_pseudo_gt(folder, max_frames):
    files = list_supported_files(folder)
    if max_frames > 0:
        files = files[:max_frames]
    stack = [load_2d(f) for f in files]
    return np.mean(np.stack(stack, axis=0), axis=0).astype(np.float32)


@torch.no_grad()
def denoise(model, z, device):
    """在 log1p 域跑一个去噪器（Denoiser 或 DenoiserWithFeats，均返回去噪图）。"""
    t = torch.from_numpy(z).float().unsqueeze(0).unsqueeze(0).to(device)
    return model(t).squeeze(0).squeeze(0).cpu().numpy()


def _norm_u8(x, vmin, vmax):
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.uint8)
    y = np.clip((x.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def save_panels(named_images, out_path, zoom_size):
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
        panels.append(p); zooms.append(z)
    canvas = Image.new("RGB", (w * len(panels), h * 2 + 24), color=(0, 0, 0))
    for i, (name, _) in enumerate(named_images):
        canvas.paste(panels[i], (i * w, 24))
        canvas.paste(zooms[i], (i * w, h + 24))
        ImageDraw.Draw(canvas).text((i * w + 5, 5), name, fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    n2n = Denoiser(input_channels=1).to(device).eval()
    print("[INFO] N2N:", load_weights_flexible(n2n, args.n2n_checkpoint, device))
    robust = DenoiserWithFeats(input_channels=1).to(device).eval()
    print("[INFO] Robust-N2N:", load_weights_flexible(robust, args.robust_checkpoint, device))

    scene_map = build_scene_index(args.data_path, args.data_subdirs, bool(args.strict_data_subdir))
    all_levels = sorted({lv for d in scene_map.values() for lv in d})
    gt_level = args.gt_level if args.gt_level > 0 else max(all_levels)
    scene_filter = set(args.scenes) if args.scenes else None
    print(f"[INFO] levels present={all_levels}; eval_level={args.eval_level}, "
          f"gt=level{gt_level}-mean, SSIM={SSIM_BACKEND}")

    rows = {"noisy": [], "n2n": [], "robust": []}
    per_scene = []
    vis_done = 0
    compare_dir = Path(args.out_dir) / "compare"

    scenes = sorted(scene_map.keys(), key=lambda s: int(s) if s.isdigit() else s)
    for scene in scenes:
        levels = scene_map[scene]
        if scene_filter is not None and scene not in scene_filter:
            continue
        if args.eval_level not in levels or gt_level not in levels:
            continue
        if args.max_scenes > 0 and len(per_scene) >= args.max_scenes:
            break

        gt_z = log1p_np(make_pseudo_gt(levels[gt_level], args.gt_frames))
        dr = float(gt_z.max() - gt_z.min()) or 1.0

        files = list_supported_files(levels[args.eval_level])
        if args.max_frames_per_scene > 0:
            files = files[:args.max_frames_per_scene]

        s_acc = {"noisy": [], "n2n": [], "robust": []}
        for fi, f in enumerate(files):
            noisy_z = log1p_np(load_2d(f))
            n2n_z = denoise(n2n, noisy_z, device)
            rob_z = denoise(robust, noisy_z, device)
            for key, img in (("noisy", noisy_z), ("n2n", n2n_z), ("robust", rob_z)):
                s_acc[key].append((psnr(img, gt_z, data_range=dr), ssim(img, gt_z, data_range=dr)))
            if vis_done < args.max_vis_scenes and fi == 0:
                save_panels(
                    [("noisy(level%d)" % args.eval_level, noisy_z), ("N2N", n2n_z),
                     ("Robust-N2N", rob_z), ("pseudo-GT(level%d)" % gt_level, gt_z)],
                    compare_dir / f"scene{scene}_frame0.png", args.zoom_size,
                )

        scene_means = {k: np.mean(v, axis=0) for k, v in s_acc.items()}
        for k in rows:
            rows[k].append(scene_means[k])
        per_scene.append({"scene": scene, **{k: scene_means[k].tolist() for k in scene_means}})
        if vis_done < args.max_vis_scenes:
            vis_done += 1

    if not per_scene:
        raise RuntimeError(f"没有可评估的场景（eval_level={args.eval_level}），检查数据/层级/筛选。")

    summary = {k: np.mean(rows[k], axis=0).tolist() for k in rows}
    print(f"\n==== OOD eval on level{args.eval_level} (vs pseudo-GT level{gt_level}, {len(per_scene)} scenes, log1p域) ====")
    print(f"{'method':>14} | {'PSNR(dB)':>9} | {'SSIM':>7}")
    print("-" * 36)
    label = {"noisy": "noisy input", "n2n": "N2N (lv234)", "robust": "Robust-N2N"}
    for k in ("noisy", "n2n", "robust"):
        print(f"{label[k]:>14} | {summary[k][0]:>9.3f} | {summary[k][1]:>7.4f}")
    gain_psnr = summary["robust"][0] - summary["n2n"][0]
    gain_ssim = summary["robust"][1] - summary["n2n"][1]
    print("-" * 36)
    print(f"{'Robust - N2N':>14} | {gain_psnr:>+9.3f} | {gain_ssim:>+7.4f}   <- OOD 泛化增益")

    out = {
        "eval_level": args.eval_level, "gt_level": gt_level, "ssim_backend": SSIM_BACKEND,
        "n_scenes": len(per_scene),
        "summary": {label[k]: {"psnr": summary[k][0], "ssim": summary[k][1]} for k in rows},
        "gain_robust_minus_n2n": {"psnr": gain_psnr, "ssim": gain_ssim},
        "per_scene": per_scene,
    }
    out_path = Path(args.out_dir) / f"metrics_level{args.eval_level}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] 指标写入 {out_path}；对比图在 {compare_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="OOD generalization eval: N2N vs Robust-N2N on held-out level.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--data_subdirs", nargs="*", default=["npy", "lbf"])
    p.add_argument("--strict_data_subdir", type=int, default=1)
    p.add_argument("--eval_level", type=int, default=1, help="留出的 OOD 评估层级（默认最噪 level1）。")
    p.add_argument("--gt_level", type=int, default=0, help="伪 GT 用的最高叠加层级；<=0 自动取最大。")
    p.add_argument("--gt_frames", type=int, default=0, help="伪 GT 多帧均值用多少帧；<=0 用全部。")
    p.add_argument("--scenes", type=str, nargs="*", default=None, help="只评估这些场景编号。")
    p.add_argument("--max_frames_per_scene", type=int, default=3)
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--max_vis_scenes", type=int, default=12)
    p.add_argument("--zoom_size", type=int, default=128)
    p.add_argument("--n2n_checkpoint", type=str, required=True)
    p.add_argument("--robust_checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="results/eval_ood_robust")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
