# -*- coding: utf-8 -*-
"""从原始散斑算少量 BFI(1/K^2) 做泛化测试，口径与 lightweight_G 的
speckle_invK2_5x5x5.py 完全一致：先对 I、I^2 做 win×win 空间盒均值(reflect)，
再对 twin 帧取时间均值，var=<I^2>-<I>^2，BFI=<I>^2/(var+eps)。

只算少量：每个 5x5xN(N=1..5) 取若干时间位置各一张；
参考两张：1x1x100（纯时间、空间最锐，用户指定）和 5x5x100（同 5×5 空间口径、苹果对苹果）。
输出 float32 npy（原始 1/K^2，与训练数据同尺度）+ png 预览。
"""
import argparse
import os
import re
import numpy as np
from PIL import Image, ImageSequence
from scipy.ndimage import uniform_filter

EPS = 1e-8


def read_f32(path):
    with Image.open(path) as im:
        try:
            im = next(ImageSequence.Iterator(im))
        except Exception:
            pass
        if im.mode in ('RGB', 'RGBA'):
            im = im.convert('L')
        a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float32, copy=False)


def tif_index(f):
    m = re.search(r'_(\d+)\.(?:tif|tiff)$', f.lower())
    return int(m.group(1)) if m else 10 ** 18


def save(arr, npy_path):
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, arr.astype(np.float32))
    lo, hi = np.percentile(arr, [1, 99])
    vv = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
    Image.fromarray((vv * 255).astype(np.uint8)).save(npy_path[:-4] + '.png')


def inv_k2_from_moments(mu, mm2):
    var = np.maximum(mm2 - mu * mu, 0.0)
    return (mu * mu / (var + EPS)).astype(np.float32)


def inv_k2(win, frame_list):
    ms = [uniform_filter(f, size=win, mode='reflect') for f in frame_list]
    m2s = [uniform_filter(f * f, size=win, mode='reflect') for f in frame_list]
    return inv_k2_from_moments(np.mean(ms, axis=0), np.mean(m2s, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True, help='存放原始散斑 tif 的文件夹')
    ap.add_argument('--out_dir', default='results/foot_bfi')
    ap.add_argument('--ext', default='.tif')
    ap.add_argument('--ref_frames', type=int, default=100, help='参考(1x1x100/5x5x100)用前多少帧')
    ap.add_argument('--centers', type=int, nargs='*', default=[10, 30, 50, 70, 90],
                    help='5x5xN 输出的时间起点（窗口 [s:s+twin]）')
    args = ap.parse_args()

    files = sorted([f for f in os.listdir(args.in_dir) if f.lower().endswith(args.ext.lower())], key=tif_index)
    if not files:
        raise RuntimeError(f'在 {args.in_dir} 没找到 {args.ext} 文件')
    files = files[:args.ref_frames]
    print(f'[INFO] 用 {args.in_dir} 的前 {len(files)} 帧')

    needed = set()
    for s in args.centers:
        for k in range(s, s + 5):
            needed.add(k)

    stash = {}
    sum_i = sum_i2 = sum_m5 = sum_m25 = None  # 1x1: I,I^2 一阶/二阶矩；5x5: box-mean 后的矩
    for k, fn in enumerate(files):
        img = read_f32(os.path.join(args.in_dir, fn))
        m5 = uniform_filter(img, size=5, mode='reflect')
        m25 = uniform_filter(img * img, size=5, mode='reflect')
        if sum_i is None:
            sum_i = np.zeros_like(img); sum_i2 = np.zeros_like(img)
            sum_m5 = np.zeros_like(img); sum_m25 = np.zeros_like(img)
        sum_i += img; sum_i2 += img * img
        sum_m5 += m5; sum_m25 += m25
        if k in needed:
            stash[k] = img
    n = float(len(files))
    print(f'[INFO] 帧尺寸 {sum_i.shape}')

    # 两张参考
    ref_t = inv_k2_from_moments(sum_i / n, sum_i2 / n)        # 1x1x100（纯时间）
    ref_s = inv_k2_from_moments(sum_m5 / n, sum_m25 / n)      # 5x5x100（同 5×5 空间口径）
    save(ref_t, os.path.join(args.out_dir, 'ref_1x1x100.npy'))
    save(ref_s, os.path.join(args.out_dir, 'ref_5x5x100.npy'))
    print(f'[INFO] ref_1x1x100 range {float(ref_t.min()):.3f}..{float(ref_t.max()):.3f}; '
          f'ref_5x5x100 range {float(ref_s.min()):.3f}..{float(ref_s.max()):.3f}')

    # 5x5x1..5
    for twin in [1, 2, 3, 4, 5]:
        for s in args.centers:
            arr = inv_k2(5, [stash[k] for k in range(s, s + twin)])
            save(arr, os.path.join(args.out_dir, f'5x5x{twin}', '0', 'npy', f'{s}.npy'))
        print(f'[INFO] done 5x5x{twin} ({len(args.centers)} frames)')

    print(f'[DONE] -> {args.out_dir}  (5x5x1..5/0/npy/*.npy + ref_1x1x100.npy + ref_5x5x100.npy)')


if __name__ == '__main__':
    main()
