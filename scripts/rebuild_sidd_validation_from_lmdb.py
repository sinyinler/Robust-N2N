"""从公开的无损 PNG LMDB 镜像重建 SIDD Validation 两个标准 MAT 文件。"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import lmdb
import numpy as np
from scipy.io import savemat
from tqdm import tqdm


INDEX_PATTERN = re.compile(r"ValidationBlocksSrgb_(\d+)\.png")


def ordered_keys(lmdb_dir: Path) -> list[str]:
    meta_path = lmdb_dir / "meta_info.txt"
    keys: list[tuple[int, str]] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        name = line.split()[0]
        match = INDEX_PATTERN.fullmatch(name)
        if match:
            keys.append((int(match.group(1)), Path(name).stem))
    keys.sort()
    if [index for index, _ in keys] != list(range(1280)):
        raise ValueError(f"{meta_path} 的 block index 不是完整的 0..1279")
    return [key for _, key in keys]


def read_lmdb(lmdb_dir: Path) -> np.ndarray:
    keys = ordered_keys(lmdb_dir)
    array = np.empty((1280, 256, 256, 3), dtype=np.uint8)
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin(write=False) as transaction:
            for index, key in enumerate(tqdm(keys, desc=lmdb_dir.name)):
                encoded = transaction.get(key.encode("ascii"))
                if encoded is None:
                    raise KeyError(f"LMDB 缺少 key={key}")
                bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None or bgr.shape != (256, 256, 3):
                    raise ValueError(f"key={key} 解码尺寸异常: {None if bgr is None else bgr.shape}")
                array[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        env.close()
    return array.reshape(40, 32, 256, 256, 3)


def main(args: argparse.Namespace) -> None:
    source = Path(args.lmdb_root).resolve()
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    noisy = read_lmdb(source / "input_crops.lmdb")
    gt = read_lmdb(source / "gt_crops.lmdb")
    noisy_path = output / "ValidationNoisyBlocksSrgb.mat"
    gt_path = output / "ValidationGtBlocksSrgb.mat"
    savemat(noisy_path, {"ValidationNoisyBlocksSrgb": noisy}, do_compression=True)
    savemat(gt_path, {"ValidationGtBlocksSrgb": gt}, do_compression=True)
    print(f"wrote {noisy_path} ({noisy_path.stat().st_size} bytes)")
    print(f"wrote {gt_path} ({gt_path.stat().st_size} bytes)")
    print(f"shape={noisy.shape}, dtype={noisy.dtype}, aligned={noisy.shape == gt.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_root", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
