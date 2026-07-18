"""Summarize the SIDD baseline/feature/feature+RTV ablation after evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.sidd_dataset import discover_sidd_pairs, load_sidd_pair
from utils.sidd import paired_bootstrap_ci


METHODS = {
    "baseline": {
        "train": "supervised_charbonnier_s42",
        "blocks": "validation_trained",
        "scene": "internal_test_scene008",
    },
    "feature": {
        "train": "supervised_feature_gaussian_s42",
        "blocks": "validation_feature_gaussian_s42",
        "scene": "internal_test_scene008_feature_gaussian_s42",
    },
    "feature_rtv": {
        "train": "supervised_feature_gaussian_rtv1e4_s42",
        "blocks": "validation_feature_gaussian_rtv1e4_s42",
        "scene": "internal_test_scene008_feature_gaussian_rtv1e4_s42",
    },
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def keyed(rows: list[dict[str, str]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[field] for field in key_fields): row for row in rows}


def method_comparison(
    rows_by_method: dict[str, list[dict[str, str]]], key_fields: tuple[str, ...]
) -> dict[str, dict]:
    aligned = {name: keyed(rows, key_fields) for name, rows in rows_by_method.items()}
    keys = sorted(set.intersection(*(set(rows) for rows in aligned.values())))
    if not keys:
        raise RuntimeError("No aligned evaluation samples were found")
    baseline = aligned["baseline"]
    report: dict[str, dict] = {}
    for method, rows in aligned.items():
        psnr = np.asarray([float(rows[key]["output_psnr"]) for key in keys], dtype=np.float64)
        ssim = np.asarray([float(rows[key]["output_ssim"]) for key in keys], dtype=np.float64)
        base_psnr = np.asarray([float(baseline[key]["output_psnr"]) for key in keys], dtype=np.float64)
        base_ssim = np.asarray([float(baseline[key]["output_ssim"]) for key in keys], dtype=np.float64)
        delta_psnr = psnr - base_psnr
        delta_ssim = ssim - base_ssim
        report[method] = {
            "count": len(keys),
            "psnr": float(psnr.mean()),
            "ssim": float(ssim.mean()),
            "delta_psnr_vs_baseline": float(delta_psnr.mean()),
            "delta_ssim_vs_baseline": float(delta_ssim.mean()),
            "delta_psnr_bootstrap95": paired_bootstrap_ci(delta_psnr.tolist()),
            "delta_ssim_bootstrap95": paired_bootstrap_ci(delta_ssim.tolist()),
            "wins_psnr_vs_baseline": int((delta_psnr > 0).sum()),
            "wins_ssim_vs_baseline": int((delta_ssim > 0).sum()),
        }
    return report


def pairwise_comparison(
    rows_by_method: dict[str, list[dict[str, str]]],
    key_fields: tuple[str, ...],
    candidate: str,
    reference: str,
) -> dict:
    candidate_rows = keyed(rows_by_method[candidate], key_fields)
    reference_rows = keyed(rows_by_method[reference], key_fields)
    keys = sorted(set(candidate_rows) & set(reference_rows))
    delta_psnr = np.asarray(
        [float(candidate_rows[key]["output_psnr"]) - float(reference_rows[key]["output_psnr"]) for key in keys]
    )
    delta_ssim = np.asarray(
        [float(candidate_rows[key]["output_ssim"]) - float(reference_rows[key]["output_ssim"]) for key in keys]
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "count": len(keys),
        "delta_psnr": float(delta_psnr.mean()),
        "delta_ssim": float(delta_ssim.mean()),
        "delta_psnr_bootstrap95": paired_bootstrap_ci(delta_psnr.tolist()),
        "delta_ssim_bootstrap95": paired_bootstrap_ci(delta_ssim.tolist()),
        "wins_psnr": int((delta_psnr > 0).sum()),
        "wins_ssim": int((delta_ssim > 0).sum()),
    }


def read_history(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_loss_curves(histories: dict[str, list[dict]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    for method, history in histories.items():
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["val"] for row in history], marker="o", markersize=2.5, label=method)
    ax.set(xlabel="Epoch", ylabel="Validation Charbonnier", title="SIDD scene 007 validation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_visual_grid(
    name: str,
    noisy_u8: np.ndarray,
    gt_u8: np.ndarray,
    image_dirs: dict[str, Path],
    path: Path,
    zoom_size: int = 256,
    panel_width: int = 480,
) -> None:
    arrays = [noisy_u8]
    for method in ("baseline", "feature", "feature_rtv"):
        arrays.append(np.asarray(Image.open(image_dirs[method] / f"{name}_output.png").convert("RGB")))
    arrays.append(gt_u8)
    labels = ["Noisy", "Baseline", "+Feature", "+Feature+RTV", "GT"]
    height, width = noisy_u8.shape[:2]
    zoom = min(zoom_size, height, width)
    top, left = max(0, (height - zoom) // 2), max(0, (width - zoom) // 2)
    panel_height = max(1, round(height * panel_width / width))
    canvas = Image.new("RGB", (panel_width * len(arrays), panel_height + panel_width + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate(zip(arrays, labels)):
        image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
        overview = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        crop = image.crop((left, top, left + zoom, top + zoom)).resize(
            (panel_width, panel_width), Image.Resampling.NEAREST
        )
        x = index * panel_width
        canvas.paste(overview, (x, 24))
        canvas.paste(crop, (x, 24 + panel_height))
        draw.text((x + 8, 5), label, fill="black")
    canvas.save(path)


def main(args: argparse.Namespace) -> None:
    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = {
        name: read_history(root / paths["train"] / "history.jsonl") for name, paths in METHODS.items()
    }
    blocks = {
        name: read_rows(root / paths["blocks"] / "per_block.csv") for name, paths in METHODS.items()
    }
    scenes = {
        name: read_rows(root / paths["scene"] / "per_image.csv") for name, paths in METHODS.items()
    }
    report = {
        "best_validation": {
            name: {
                "epoch": min(history, key=lambda row: row["val"])["epoch"],
                "loss": min(row["val"] for row in history),
            }
            for name, history in histories.items()
        },
        "public_validation_blocks": method_comparison(blocks, ("image", "block")),
        "internal_scene008": method_comparison(scenes, ("name",)),
        "pairwise": {
            "public_feature_vs_baseline": pairwise_comparison(
                blocks, ("image", "block"), "feature", "baseline"
            ),
            "public_feature_rtv_vs_feature": pairwise_comparison(
                blocks, ("image", "block"), "feature_rtv", "feature"
            ),
            "scene008_feature_vs_baseline": pairwise_comparison(
                scenes, ("name",), "feature", "baseline"
            ),
            "scene008_feature_rtv_vs_feature": pairwise_comparison(
                scenes, ("name",), "feature_rtv", "feature"
            ),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    flat_rows = []
    for split in ("public_validation_blocks", "internal_scene008"):
        for method, values in report[split].items():
            flat_rows.append({"split": split, "method": method, **values})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        scalar_fields = [key for key, value in flat_rows[0].items() if not isinstance(value, (list, tuple))]
        writer = csv.DictWriter(handle, fieldnames=scalar_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    save_loss_curves(histories, out_dir / "validation_loss_comparison.png")

    pairs = {pair.name: pair for pair in discover_sidd_pairs(args.data_root, ("008",))}
    image_dirs = {name: root / paths["scene"] / "images" for name, paths in METHODS.items()}
    for name in args.visual_names:
        if name not in pairs:
            raise KeyError(f"Unknown scene008 pair: {name}")
        noisy_u8, gt_u8 = load_sidd_pair(pairs[name])
        save_visual_grid(name, noisy_u8, gt_u8, image_dirs, out_dir / f"{name}_comparison.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="results/sidd")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="results/sidd/ablation_comparison")
    parser.add_argument(
        "--visual_names",
        nargs="+",
        default=[
            "0170_008_N6_01600_00800_4400_L",
            "0180_008_GP_00100_00100_5500_N",
            "0188_008_IP_00100_00100_3200_N",
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
