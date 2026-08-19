#!/usr/bin/env python3
"""Compare SIDD-Small and SIDD-Medium feature/RTV ablations.

The script uses paired units wherever possible.  In particular, Small's single
capture is aligned with Medium capture 010, whose noisy/GT images are identical.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sidd"


RUNS = {
    "small_feature": {
        "label": "Small + feature",
        "validation": RESULTS / "validation_feature_gaussian_s42",
        "scene": RESULTS / "internal_test_scene008_feature_gaussian_s42",
        "history": RESULTS / "scene_split_feature_gaussian_s42" / "history.jsonl",
    },
    "medium_feature": {
        "label": "Medium + feature",
        "validation": RESULTS / "validation_medium_feature_gaussian_s42",
        "scene": RESULTS / "internal_test_scene008_medium_feature_gaussian_s42",
        "history": RESULTS / "medium_scene_split_feature_gaussian_s42" / "history.jsonl",
    },
    "small_rtv": {
        "label": "Small + feature + RTV",
        "validation": RESULTS / "validation_feature_gaussian_rtv1e4_s42",
        "scene": RESULTS / "internal_test_scene008_feature_gaussian_rtv1e4_s42",
        "history": RESULTS / "scene_split_feature_gaussian_rtv1e4_s42" / "history.jsonl",
    },
    "medium_rtv": {
        "label": "Medium + feature + RTV",
        "validation": RESULTS / "validation_medium_feature_gaussian_rtv1e4_s42",
        "scene": RESULTS / "internal_test_scene008_medium_feature_gaussian_rtv1e4_s42",
        "history": RESULTS / "medium_scene_split_feature_gaussian_rtv1e4_s42" / "history.jsonl",
    },
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_history(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def paired_bootstrap(values: np.ndarray, seed: int = 42, samples: int = 20_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 1_000
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def compare_rows(rows_a, rows_b, keys, label_a, label_b):
    map_a = {tuple(row[key] for key in keys): row for row in rows_a}
    map_b = {tuple(row[key] for key in keys): row for row in rows_b}
    common = sorted(set(map_a) & set(map_b))
    result = {
        "a": label_a,
        "b": label_b,
        "paired_count": len(common),
        "definition": "difference = b - a",
    }
    for metric in ("output_psnr", "output_ssim"):
        diff = np.array([float(map_b[key][metric]) - float(map_a[key][metric]) for key in common])
        result[metric] = {
            "mean_difference": float(diff.mean()),
            "bootstrap95": paired_bootstrap(diff),
            "b_wins": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "a_wins": int((diff < 0).sum()),
        }
    return result


def medium_capture(rows, suffix="_010"):
    converted = []
    for row in rows:
        if not row["name"].endswith(suffix):
            continue
        item = dict(row)
        item["name"] = item["name"][: -len(suffix)]
        converted.append(item)
    return converted


def summary_rows():
    rows = []
    for key, run in RUNS.items():
        val = read_json(run["validation"] / "summary.json")
        scene = read_json(run["scene"] / "summary.json")
        rows.append(
            {
                "run": key,
                "label": run["label"],
                "validation_psnr": val["output_psnr"],
                "validation_ssim": val["output_ssim"],
                "scene008_count": scene["pair_count"],
                "scene008_psnr": scene["output_psnr"],
                "scene008_ssim": scene["output_ssim"],
                "scene008_psnr_gain": scene["delta_psnr"],
                "scene008_ssim_gain": scene["delta_ssim"],
            }
        )
    return rows


def make_loss_curve(out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for key in ("medium_feature", "medium_rtv"):
        history = read_history(RUNS[key]["history"])
        ax.plot([x["epoch"] for x in history], [x["val"] for x in history], marker="o", ms=3, label=RUNS[key]["label"])
    ax.set(xlabel="Epoch", ylabel="Validation Charbonnier loss", title="SIDD-Medium validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "medium_validation_loss.png")
    plt.close(fig)


def choose_crop(noisy_path: Path, gt_path: Path, crop: int = 512):
    with Image.open(noisy_path) as image:
        width, height = image.size
        thumb_size = (max(1, width // 8), max(1, height // 8))
        noisy = np.asarray(image.convert("RGB").resize(thumb_size, Image.Resampling.BILINEAR), dtype=np.float32)
    with Image.open(gt_path) as image:
        gt = np.asarray(image.convert("RGB").resize(thumb_size, Image.Resampling.BILINEAR), dtype=np.float32)
    error = np.abs(noisy - gt).mean(axis=2)
    gray = gt.mean(axis=2)
    grad = np.zeros_like(gray)
    grad[:, 1:] += np.abs(np.diff(gray, axis=1))
    grad[1:, :] += np.abs(np.diff(gray, axis=0))
    window = max(3, crop // 8)
    score = uniform_filter(error, window, mode="nearest") + 0.20 * uniform_filter(grad, window, mode="nearest")
    margin = window // 2 + 2
    score[:margin] = -np.inf
    score[-margin:] = -np.inf
    score[:, :margin] = -np.inf
    score[:, -margin:] = -np.inf
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    left = int(np.clip(cx * 8 - crop // 2, 0, width - crop))
    top = int(np.clip(cy * 8 - crop // 2, 0, height - crop))
    return left, top, left + crop, top + crop


def image_crop(path: Path, box):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").crop(box))


def make_visual_grids(data_root: Path, out_dir: Path):
    examples = [
        "0170_008_N6_01600_00800_4400_L",
        "0180_008_GP_00100_00100_5500_N",
        "0188_008_IP_00100_00100_3200_N",
    ]
    for name in examples:
        instance = data_root / name
        prefix = name.split("_", 1)[0]
        noisy = instance / f"{prefix}_NOISY_SRGB_010.PNG"
        gt = instance / f"{prefix}_GT_SRGB_010.PNG"
        sources = [
            ("Noisy", noisy),
            ("Small + feature", RUNS["small_feature"]["scene"] / "images" / f"{name}_output.png"),
            ("Medium + feature", RUNS["medium_feature"]["scene"] / "images" / f"{name}_010_output.png"),
            ("Small + feature + RTV", RUNS["small_rtv"]["scene"] / "images" / f"{name}_output.png"),
            ("Medium + feature + RTV", RUNS["medium_rtv"]["scene"] / "images" / f"{name}_010_output.png"),
            ("GT", gt),
        ]
        missing = [str(path) for _, path in sources if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing visual inputs: " + ", ".join(missing))
        box = choose_crop(noisy, gt)
        fig, axes = plt.subplots(1, len(sources), figsize=(18, 3.5), dpi=160)
        for axis, (title, path) in zip(axes, sources):
            axis.imshow(image_crop(path, box))
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        fig.suptitle(f"{name} capture 010; crop={box}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"visual_{name}.png", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"E:\SIDD\SIDD_Medium_Srgb\mnt\d\SIDD_Medium_Srgb\Data"),
    )
    parser.add_argument("--out-dir", type=Path, default=RESULTS / "medium_ablation_comparison")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = summary_rows()
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    block = {key: read_csv(run["validation"] / "per_block.csv") for key, run in RUNS.items()}
    scene = {key: read_csv(run["scene"] / "per_image.csv") for key, run in RUNS.items()}
    comparisons = {
        "validation_medium_vs_small_feature": compare_rows(block["small_feature"], block["medium_feature"], ["image", "block"], "small_feature", "medium_feature"),
        "validation_medium_vs_small_rtv": compare_rows(block["small_rtv"], block["medium_rtv"], ["image", "block"], "small_rtv", "medium_rtv"),
        "validation_medium_rtv_vs_feature": compare_rows(block["medium_feature"], block["medium_rtv"], ["image", "block"], "medium_feature", "medium_rtv"),
        "scene010_medium_vs_small_feature": compare_rows(scene["small_feature"], medium_capture(scene["medium_feature"]), ["name"], "small_feature", "medium_feature_010"),
        "scene010_medium_vs_small_rtv": compare_rows(scene["small_rtv"], medium_capture(scene["medium_rtv"]), ["name"], "small_rtv", "medium_rtv_010"),
        "scene010_medium_rtv_vs_feature": compare_rows(medium_capture(scene["medium_feature"]), medium_capture(scene["medium_rtv"]), ["name"], "medium_feature_010", "medium_rtv_010"),
        "scene011_medium_rtv_vs_feature": compare_rows(medium_capture(scene["medium_feature"], "_011"), medium_capture(scene["medium_rtv"], "_011"), ["name"], "medium_feature_011", "medium_rtv_011"),
        "scene40_medium_rtv_vs_feature": compare_rows(scene["medium_feature"], scene["medium_rtv"], ["name"], "medium_feature", "medium_rtv"),
    }
    failure_cases = {
        key: [
            {
                "name": row["name"],
                "noisy_psnr": float(row["noisy_psnr"]),
                "output_psnr": float(row["output_psnr"]),
                "delta_psnr": float(row["delta_psnr"]),
                "delta_ssim": float(row["delta_ssim"]),
            }
            for row in scene[key]
            if float(row["delta_psnr"]) < 0
        ]
        for key in ("medium_feature", "medium_rtv")
    }
    payload = {"summaries": rows, "paired_comparisons": comparisons, "negative_psnr_gain_cases": failure_cases}
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_loss_curve(args.out_dir)
    make_visual_grids(args.data_root, args.out_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
