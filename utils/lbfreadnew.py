from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# 文件头标记：与 MATLAB 版本中的 0x424C 判断一致。
TYPE_MAGIC = 0x424C

# 头部偏移：与原 MATLAB 程序完全保持一致。
WIDTH_OFFSET = 18
HEIGHT_OFFSET = 22
FLOAT_DATA_OFFSET = 136
UINT16_DATA_OFFSET = 160


def _read_exact(handle, size: int) -> bytes:
    """从二进制文件中读取固定字节数，长度不够时直接报错。"""
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"文件内容不足，期望读取 {size} 字节，实际只读到 {len(data)} 字节。")
    return data


def _read_int32_le(handle) -> int:
    """按小端 int32 读取一个整数。Windows 下的 MATLAB 默认也是这个字节序。"""
    return int(np.frombuffer(_read_exact(handle, 4), dtype="<i4", count=1)[0])


def lbfreadnew(path: str | Path) -> np.ndarray:
    """
    读取 .lbf 文件并返回二维矩阵。

    这段逻辑是对 lbfreadnew.m 的 Python 等价实现：
    1. 读取文件头前 2 字节作为类型标记；
    2. 从偏移 18 和 22 读取宽度、高度；
    3. 若类型标记为 0x424C，则从偏移 136 按 float32 读取；
    4. 否则从偏移 160 按 uint16 读取；
    5. 最后整理成 height x width 的二维数组。

    返回值统一转成 float64，这样更接近 MATLAB fread 的默认输出行为。
    """
    file_path = Path(path)

    with file_path.open("rb") as handle:
        file_type = int(np.frombuffer(_read_exact(handle, 2), dtype="<u2", count=1)[0])

        handle.seek(WIDTH_OFFSET)
        width = _read_int32_le(handle)
        height = _read_int32_le(handle)

        if width <= 0 or height <= 0:
            raise ValueError(f"读取到非法尺寸：width={width}, height={height}")

        if file_type == TYPE_MAGIC:
            data_offset = FLOAT_DATA_OFFSET
            source_dtype = np.dtype("<f4")
        else:
            data_offset = UINT16_DATA_OFFSET
            source_dtype = np.dtype("<u2")

        handle.seek(data_offset)
        element_count = width * height
        raw = np.fromfile(handle, dtype=source_dtype, count=element_count)

    if raw.size != element_count:
        raise ValueError(
            f"数据区长度不足：期望读取 {element_count} 个元素，实际只读到 {raw.size} 个元素。"
        )

    # MATLAB 的写法是 fread(..., [width height]) 再转置。
    # 等价到 Python，就是按顺序整理成 height x width 的二维矩阵。
    matrix = raw.reshape((height, width))

    # MATLAB fread 默认输出 double，这里主动转成 float64，便于和原程序结果保持一致。
    return matrix.astype(np.float64, copy=False)


def save_preview_image(data: np.ndarray, output_path: str | Path) -> Path:
    """
    将二维矩阵线性拉伸到 0~255，并保存为灰度 PNG 预览图。
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("保存预览图需要先安装 Pillow：pip install pillow") from exc

    output = Path(output_path)
    data64 = np.asarray(data, dtype=np.float64)

    data_min = float(np.min(data64))
    data_max = float(np.max(data64))

    if data_max > data_min:
        scaled = np.rint((data64 - data_min) / (data_max - data_min) * 255.0).astype(np.uint8)
    else:
        scaled = np.zeros(data64.shape, dtype=np.uint8)

    Image.fromarray(scaled, mode="L").save(output)
    return output


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取 .lbf 文件并输出矩阵信息。")
    parser.add_argument(
        "files",
        nargs="*",
        help="要读取的 .lbf 文件路径。不传时，默认读取当前目录下所有 .lbf 文件。",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="为每个 .lbf 文件额外保存一张灰度 PNG 预览图。",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.files:
        targets = [Path(item) for item in args.files]
    else:
        targets = sorted(Path.cwd().glob("*.lbf"))

    if not targets:
        parser.error("没有找到可读取的 .lbf 文件。")

    for file_path in targets:
        data = lbfreadnew(file_path)

        print(f"已读取: {file_path}")
        print(f"矩阵大小: {data.shape[0]} x {data.shape[1]}")
        print(f"数据类型: {data.dtype}")

        if args.save_preview:
            preview_path = file_path.with_name(f"{file_path.stem}_preview_py.png")
            save_preview_image(data, preview_path)
            print(f"预览图已保存: {preview_path}")

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
