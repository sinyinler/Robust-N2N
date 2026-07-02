from __future__ import annotations

"""诊断 NTN 的翻译器 T 到底有没有在「翻译噪声」，定位 NTN≈N2N 的原因。

在留出的 OOD 层级（默认 level1）上，对每个场景的若干帧回答三件事：

1) T 是否起作用：对比
   - D'(I)      —— 不翻译，直接把噪声图喂给盲高斯专家 D'
   - D'(T(I))   —— 完整 NTN
   若两者输出几乎一样（PSNR_self 很高），说明 T ≈ 恒等映射、没学到东西。

2) 翻译后的噪声是否更「高斯 / 更白」（论文翻译奏效的硬指标）：
   以同场景高叠加层多帧均值为干净参考 Ĉ，比较
   - 真实噪声      n_real  = I    - Ĉ
   - 翻译后噪声    n_trans = T(I) - Ĉ
   指标：std（幅度）、lag-1 空间自相关（→0 表示更白）、谱平坦度（→1 表示更白）、
        峰度 excess kurtosis（→0 表示更高斯）。

3) GIBlock 注入强度：打印每个 block 学到的 noise_scale 绝对值（初值 0.1）。

输出：终端汇总表 + results/diag/noise_panels/scene*.png（I / T(I) / |n_real| / |n_trans|）
     + results/diag/diagnose.json。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.discovery import list_supported_files, load_2d
from eval_ood import build_scene_index, log1p_np, make_pseudo_gt, save_panels
from models.denoiser import Denoiser
from models.ntn import NoiseTranslator
from utils.checkpoint import load_weights_flexible
from utils.metrics import center_crop_to_match, psnr, ssim_simple


def center_crop(x: np.ndarray, c: int) -> np.ndarray:
    if c <= 0 or (x.shape[0] <= c and x.shape[1] <= c):
        return x
    t, l = (x.shape[0] - c) // 2, (x.shape[1] - c) // 2
    return x[t:t + c, l:l + c]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
    return float((a * b).sum() / denom)


def lag1_autocorr(n: np.ndarray) -> float:
    """lag-1 空间自相关（横、纵平均）。白噪声≈0，空间相关噪声为正。"""
    h = pearson(n[:, :-1], n[:, 1:])
    v = pearson(n[:-1, :], n[1:, :])
    return 0.5 * (h + v)


def spectral_flatness(n: np.ndarray) -> float:
    """谱平坦度（Wiener entropy）。白噪声→1，低频占优的相关噪声→<<1。"""
    f = np.abs(np.fft.fft2(n - n.mean())) ** 2
    f = f.ravel() + 1e-12
    return float(np.exp(np.mean(np.log(f))) / np.mean(f))


def excess_kurtosis(n: np.ndarray) -> float:
    x = n.ravel().astype(np.float64)
    x = x - x.mean()
    var = float((x * x).mean()) or 1.0
    return float((x ** 4).mean() / (var ** 2) - 3.0)


def noise_stats(n: np.ndarray) -> dict:
    return {
        "std": float(n.std()),
        "lag1_autocorr": lag1_autocorr(n),
        "spectral_flatness": spectral_flatness(n),
        "excess_kurtosis": excess_kurtosis(n),
    }


@torch.no_grad()
def fwd(model, z: np.ndarray, device) -> np.ndarray:
    t = torch.from_numpy(z).float().unsqueeze(0).unsqueeze(0).to(device)
    return model(t).squeeze(0).squeeze(0).cpu().numpy()


@torch.no_grad()
def translate(translator, z: np.ndarray, device) -> np.ndarray:
    t = torch.from_numpy(z).float().unsqueeze(0).unsqueeze(0).to(device)
    return translator(t).squeeze(0).squeeze(0).cpu().numpy()


def save_abs(x: np.ndarray, path: Path, scale: float = 1.0) -> None:
    a = np.abs(x) * scale
    vmax = float(np.percentile(a, 99)) or 1.0
    u8 = np.clip(a / vmax, 0, 1) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8.astype(np.uint8), mode="L").save(path)


def report_giblock(translator) -> dict:
    """打印并返回每个 GIBlock 的 noise_scale 幅度（初值 0.1）。"""
    stats = {}
    for name, p in translator.named_parameters():
        if name.endswith("noise_scale"):
            stats[name] = float(p.detach().abs().mean().cpu())
    print("\n[GIBlock noise_scale |mean abs| per block] (初值 0.1)")
    for k, v in stats.items():
        print(f"  {k:55s} {v:.4f}")
    if stats:
        print(f"  -> 全部平均 {np.mean(list(stats.values())):.4f}")
    return stats


def main(args) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    expert = Denoiser(input_channels=1).to(device).eval()
    print("[INFO] D':", load_weights_flexible(expert, args.gaussian_expert_checkpoint, device))
    n2n = Denoiser(input_channels=1).to(device).eval()
    print("[INFO] N2N:", load_weights_flexible(n2n, args.n2n_checkpoint, device))
    translator = NoiseTranslator(
        input_channels=1, width=args.width, middle_blocks=args.middle_blocks,
        inject_sigma=args.inject_sigma, residual_scale=args.residual_scale,
    ).to(device).eval()
    print("[INFO] T:", load_weights_flexible(translator, args.translator_checkpoint, device))

    report_giblock(translator)

    scene_map = build_scene_index(args.data_path, args.data_subdirs, bool(args.strict_data_subdir))
    all_levels = sorted({lv for d in scene_map.values() for lv in d})
    gt_level = args.gt_level if args.gt_level > 0 else max(all_levels)

    acc = {"real": [], "trans": [],
           "psnr_n2n": [], "psnr_Donly": [], "psnr_ntn": [],
           "moveT_rel": [], "self_psnr_Donly_vs_ntn": []}
    panel_dir = Path(args.out_dir) / "noise_panels"
    n_scenes = 0

    for scene in sorted(scene_map, key=lambda s: int(s) if s.isdigit() else s):
        levels = scene_map[scene]
        if args.eval_level not in levels or gt_level not in levels:
            continue
        if args.max_scenes > 0 and n_scenes >= args.max_scenes:
            break
        gt_z = log1p_np(make_pseudo_gt(levels[gt_level], args.gt_frames))
        gt_c = center_crop(gt_z, args.crop)
        dr = float(gt_c.max() - gt_c.min()) or 1.0

        files = list_supported_files(levels[args.eval_level])
        if args.max_frames_per_scene > 0:
            files = files[:args.max_frames_per_scene]

        for fi, f in enumerate(files):
            I = log1p_np(load_2d(f))
            TI = translate(translator, I, device)
            out_n2n = fwd(n2n, I, device)
            out_Donly = fwd(expert, I, device)
            out_ntn = fwd(expert, TI, device)

            Ic, TIc = center_crop(I, args.crop), center_crop(TI, args.crop)
            n_real = Ic - gt_c
            n_trans = TIc - gt_c
            acc["real"].append(noise_stats(n_real))
            acc["trans"].append(noise_stats(n_trans))
            acc["moveT_rel"].append(float(np.linalg.norm(TIc - Ic) / (np.linalg.norm(Ic) or 1.0)))

            acc["psnr_n2n"].append(psnr(out_n2n, gt_z, data_range=dr))
            acc["psnr_Donly"].append(psnr(out_Donly, gt_z, data_range=dr))
            acc["psnr_ntn"].append(psnr(out_ntn, gt_z, data_range=dr))
            acc["self_psnr_Donly_vs_ntn"].append(psnr(out_Donly, out_ntn, data_range=dr))

            if n_scenes < args.max_vis_scenes and fi == 0:
                save_panels(
                    [("I (level%d)" % args.eval_level, Ic), ("T(I)", TIc),
                     ("|n_real|x%g" % args.noise_vis_scale, np.abs(n_real) * args.noise_vis_scale),
                     ("|n_trans|x%g" % args.noise_vis_scale, np.abs(n_trans) * args.noise_vis_scale)],
                    panel_dir / f"scene{scene}_frame0.png", min(160, args.crop),
                )
        n_scenes += 1

    if n_scenes == 0:
        raise RuntimeError("没有同时含 eval_level 和 gt_level 的场景。")

    def avg(key, sub=None):
        if sub is None:
            return float(np.mean(acc[key]))
        return float(np.mean([d[sub] for d in acc[key]]))

    print(f"\n==== Translation diagnosis on level{args.eval_level} (vs GT level{gt_level}, {n_scenes} scenes) ====")
    print("\n[1] T 是否改变了去噪结果？(self-PSNR 越高 = D'(I) 和 D'(T(I)) 越像 = T 越没用)")
    print(f"    PSNR  D'(I) no-T   = {avg('psnr_Donly'):.3f} dB")
    print(f"    PSNR  D'(T(I)) NTN = {avg('psnr_ntn'):.3f} dB")
    print(f"    PSNR  N2N baseline = {avg('psnr_n2n'):.3f} dB")
    print(f"    self-PSNR D'(I) vs D'(T(I)) = {avg('self_psnr_Donly_vs_ntn'):.2f} dB  (>40 基本等同)")
    print(f"    T 对输入的相对改动 ||T(I)-I||/||I|| = {avg('moveT_rel'):.4f}")

    print("\n[2] 翻译前后噪声统计 (real=I-GT, trans=T(I)-GT)")
    print(f"    {'metric':>18} | {'real':>10} | {'trans':>10} | 期望方向")
    print("    " + "-" * 58)
    print(f"    {'std (幅度)':>18} | {avg('real','std'):>10.4f} | {avg('trans','std'):>10.4f} |")
    print(f"    {'lag1 自相关':>18} | {avg('real','lag1_autocorr'):>10.4f} | {avg('trans','lag1_autocorr'):>10.4f} | 越小越白(→0)")
    print(f"    {'谱平坦度':>18} | {avg('real','spectral_flatness'):>10.4f} | {avg('trans','spectral_flatness'):>10.4f} | 越大越白(→1)")
    print(f"    {'excess kurtosis':>18} | {avg('real','excess_kurtosis'):>10.4f} | {avg('trans','excess_kurtosis'):>10.4f} | 越接近0越高斯")

    out = {
        "eval_level": args.eval_level, "gt_level": gt_level, "n_scenes": n_scenes,
        "psnr": {"N2N": avg("psnr_n2n"), "Donly_noT": avg("psnr_Donly"), "NTN": avg("psnr_ntn")},
        "self_psnr_Donly_vs_ntn": avg("self_psnr_Donly_vs_ntn"),
        "moveT_rel": avg("moveT_rel"),
        "noise_real": {k: avg("real", k) for k in ("std", "lag1_autocorr", "spectral_flatness", "excess_kurtosis")},
        "noise_trans": {k: avg("trans", k) for k in ("std", "lag1_autocorr", "spectral_flatness", "excess_kurtosis")},
    }
    out_path = Path(args.out_dir) / "diagnose.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] 写入 {out_path}；噪声可视化在 {panel_dir}")
    print("\n判读提示：")
    print("  - 若 self-PSNR 很高(>40)且 moveT 很小 → T 近似恒等，没在翻译（问题在 T）。")
    print("  - 若 trans 的 lag1↓、谱平坦度↑、kurtosis→0（比 real 明显更白更高斯）→ T 确实在翻译，")
    print("    但 NTN 仍≈N2N，则瓶颈在 D'（被 N2N 目标锚死、且会磨血管）→ 该换多帧均值目标重训。")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose whether the noise translator T actually translates noise.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--data_subdirs", nargs="*", default=["npy", "lbf"])
    p.add_argument("--strict_data_subdir", type=int, default=1)
    p.add_argument("--eval_level", type=int, default=1)
    p.add_argument("--gt_level", type=int, default=0)
    p.add_argument("--gt_frames", type=int, default=0)
    p.add_argument("--crop", type=int, default=512, help="统计/出图用的中心裁剪边长。")
    p.add_argument("--max_frames_per_scene", type=int, default=2)
    p.add_argument("--max_scenes", type=int, default=10)
    p.add_argument("--max_vis_scenes", type=int, default=6)
    p.add_argument("--noise_vis_scale", type=float, default=5.0)
    p.add_argument("--n2n_checkpoint", type=str, required=True)
    p.add_argument("--translator_checkpoint", type=str, required=True)
    p.add_argument("--gaussian_expert_checkpoint", type=str, required=True)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--middle_blocks", type=int, default=2)
    p.add_argument("--inject_sigma", type=float, default=1.0)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--out_dir", type=str, default="results/diag")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
