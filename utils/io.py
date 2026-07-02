from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from utils.lbfreadnew import lbfreadnew


SUPPORTED_INPUT_EXTS = (".npy", ".lbf", ".tif", ".tiff", ".png", ".jpg", ".jpeg")


def natural_sort_key(name: str):
    stem, ext = os.path.splitext(name.lower())
    parts = re.findall(r"\d+|[a-z]+", stem)
    key = [(0, int(part)) if part.isdigit() else (1, part) for part in parts]
    key.append((2, ext))
    return key


def read_image_any(path: str | Path) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif ext == ".lbf":
        arr = lbfreadnew(str(path))
    else:
        with Image.open(path) as img:
            try:
                img = next(ImageSequence.Iterator(img))
            except Exception:
                pass
            if img.mode in ("RGB", "RGBA"):
                img = img.convert("L")
            arr = np.asarray(img)

    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got {arr.shape} from {path}")
    return arr.astype(np.float32, copy=False)


def save_npy_and_png(array: np.ndarray, npy_path: str | Path, png_path: str | Path) -> None:
    npy_path = Path(npy_path)
    png_path = Path(png_path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, array.astype(np.float32, copy=False))

    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, neginf=0.0)
    vmin = float(np.percentile(arr, 1))
    vmax = float(np.percentile(arr, 99))
    if vmax <= vmin:
        out = np.zeros_like(arr, dtype=np.uint8)
    else:
        out = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        out = (out * 255.0).astype(np.uint8)
    Image.fromarray(out, mode="L").save(png_path)


def list_inputs(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTS]
    return sorted(files, key=lambda p: natural_sort_key(p.name))
