import os
import random
import re

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset

from utils.lbfreadnew import lbfreadnew
from utils.monotonic_vst import load_vst_lut, vst_forward_torch


SUPPORTED_ARRAY_EXTS = (".npy", ".lbf")
DEFAULT_DATA_SUBDIRS = ("npy", "lbf")
BOXCOX_LAM = -0.15
BOXCOX_EPS = 1e-6
LAMBDA_MIN = -0.3
LAMBDA_MAX = 0.2


def log1p_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


def boxcox_1p_torch(x: torch.Tensor, lam: float = BOXCOX_LAM, eps: float = BOXCOX_EPS) -> torch.Tensor:
    u = torch.clamp(x, min=0.0) + 1.0 + eps
    if abs(lam) < 1e-12:
        return torch.log(u)
    return (torch.pow(u, lam) - 1.0) / lam


def lambda_condition_value(lam: float, lambda_min: float = LAMBDA_MIN, lambda_max: float = LAMBDA_MAX) -> float:
    if abs(lambda_max - lambda_min) < 1e-12:
        return float(lam)
    return float(2.0 * (lam - lambda_min) / (lambda_max - lambda_min) - 1.0)


def _normalize_lambda_candidates(lambda_candidates) -> tuple[float, ...] | None:
    if lambda_candidates is None:
        return None
    values = tuple(float(item) for item in lambda_candidates)
    return values if values else None


def _sample_lambda(
    fixed_lambda: float,
    lambda_conditioned: bool,
    lambda_min: float,
    lambda_max: float,
    lambda_candidates: tuple[float, ...] | None,
) -> float:
    if not lambda_conditioned:
        return float(fixed_lambda)
    if lambda_candidates is not None:
        return float(random.choice(lambda_candidates))
    return float(random.uniform(lambda_min, lambda_max))


def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Image size ({h}, {w}) is smaller than crop size {crop_size}")

    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h:start_h + crop_size, start_w:start_w + crop_size]


def _normalize_crop_size(crop_size: int | None) -> int | None:
    # crop_size<=0 is the explicit "use original image size" mode for new data.
    if crop_size is None or int(crop_size) <= 0:
        return None
    return int(crop_size)


def _normalize_preferred_subdirs(npy_folder_name, include_defaults: bool = True) -> tuple[str, ...]:
    if isinstance(npy_folder_name, (list, tuple)):
        names = [str(name).strip() for name in npy_folder_name if str(name).strip()]
    else:
        names = [str(npy_folder_name).strip()] if str(npy_folder_name).strip() else []

    if include_defaults:
        for default_name in DEFAULT_DATA_SUBDIRS:
            if default_name not in names:
                names.append(default_name)

    return tuple(names)


def _natural_sort_key(name: str):
    """
    Split the stem into text / integer chunks and ignore separator characters
    such as '-' and '_' so names like 2-11_9.lbf and 2-11_10.lbf are ordered
    numerically instead of lexicographically.
    """
    stem, ext = os.path.splitext(name.lower())
    parts = re.findall(r"\d+|[a-z]+", stem)
    key = [
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
    ]
    key.append((-1, ""))
    key.append((2, ext))
    return key


def _list_supported_files(folder_path: str) -> list[str]:
    if not os.path.isdir(folder_path):
        return []

    files = []
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in SUPPORTED_ARRAY_EXTS:
            files.append(name)
    return sorted(files, key=_natural_sort_key)


def _resolve_data_dir(base_dir: str, preferred_subdirs: tuple[str, ...], strict_subdir: bool = False) -> str | None:
    for subdir_name in preferred_subdirs:
        candidate = os.path.join(base_dir, subdir_name)
        if _list_supported_files(candidate):
            return candidate

    # In strict mode we only accept the requested nested folder.  This prevents
    # silently falling back to the legacy "npy" directory when the new data is
    # missing for one sequence.
    if strict_subdir:
        return None

    if _list_supported_files(base_dir):
        return base_dir

    child_candidates = []
    for name in sorted(os.listdir(base_dir)):
        candidate = os.path.join(base_dir, name)
        if os.path.isdir(candidate) and _list_supported_files(candidate):
            child_candidates.append(candidate)

    if len(child_candidates) == 1:
        return child_candidates[0]

    return None


def _resolve_data_dirs(base_dir: str, preferred_subdirs: tuple[str, ...], strict_subdir: bool = False) -> list[str]:
    """Return every usable sequence data directory below ``base_dir``.

    The original training data can provide both nested ``npy`` folders and
    direct ``.lbf`` files, for example ``mix/0/npy`` plus ``mix/0/2_1.lbf``.
    We keep each concrete folder as a separate dataset so noisy/noisy pairing
    remains inside one folder and never crosses sequence indices.
    """
    data_dirs = []

    if not strict_subdir and _list_supported_files(base_dir):
        # Old lbf data is often stored directly under mix/<index>, without a
        # separate lbf subfolder.  Treat mix/<index> itself as one sequence.
        data_dirs.append(base_dir)

    for subdir_name in preferred_subdirs:
        candidate = os.path.join(base_dir, subdir_name)
        if _list_supported_files(candidate):
            data_dirs.append(candidate)

    if data_dirs or strict_subdir:
        return data_dirs

    if _list_supported_files(base_dir):
        return [base_dir]

    child_candidates = []
    for name in sorted(os.listdir(base_dir)):
        candidate = os.path.join(base_dir, name)
        if os.path.isdir(candidate) and _list_supported_files(candidate):
            child_candidates.append(candidate)

    if len(child_candidates) == 1:
        return child_candidates

    return []


def _load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".lbf":
        arr = lbfreadnew(path)
    else:
        raise ValueError(f"Unsupported training file type: {path}")

    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {arr.shape} from {path}")

    return arr.astype(np.float32, copy=False)


def _load_array_shape(path: str) -> tuple[int, int]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = arr.shape
    else:
        shape = _load_array(path).shape

    if len(shape) == 3:
        if shape[0] == 1:
            shape = shape[1:]
        elif shape[-1] == 1:
            shape = shape[:-1]

    if len(shape) != 2:
        raise ValueError(f"Expected a 2D array, but got shape {shape} from {path}")

    return int(shape[0]), int(shape[1])


class SpeckleN2NLogDataset(Dataset):
    def __init__(
        self,
        root_dir,
        crop_size=512,
        intervals=None,
        boxcox_lam: float = BOXCOX_LAM,
        boxcox_eps: float = BOXCOX_EPS,
        lambda_conditioned: bool = False,
        lambda_min: float = LAMBDA_MIN,
        lambda_max: float = LAMBDA_MAX,
        lambda_candidates=None,
        intensity_transform: str = "log1p",
        vst_lut: str | None = None,
        npy_folder_name="npy",
        strict_data_subdir: bool = False,
        data_index_min: int | None = None,
        data_index_max: int | None = None,
        include_levels: tuple[int, ...] | None = None,
    ):
        if intervals is None:
            intervals = [5, 7, 9]
        self.samples = create_train_dataset(
            root_dir,
            intervals,
            crop_size=crop_size,
            npy_folder_name=npy_folder_name,
            strict_data_subdir=strict_data_subdir,
            data_index_min=data_index_min,
            data_index_max=data_index_max,
            include_levels=include_levels,
        )
        self.intensity_transform = str(intensity_transform).lower()
        if self.intensity_transform not in {"log1p", "boxcox", "learned_vst"}:
            raise ValueError("intensity_transform must be 'log1p', 'boxcox', or 'learned_vst'")

        self.boxcox_lam = float(boxcox_lam)
        self.boxcox_eps = float(boxcox_eps)
        self.lambda_conditioned = bool(lambda_conditioned) and self.intensity_transform == "boxcox"
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.lambda_candidates = _normalize_lambda_candidates(lambda_candidates)
        self.vst_y_values = None
        self.vst_f_values = None
        if self.intensity_transform == "learned_vst":
            if not vst_lut:
                raise ValueError("--vst_lut is required when intensity_transform='learned_vst'")
            lut = load_vst_lut(vst_lut)
            # Store CPU tensors. DataLoader workers apply this one-dimensional
            # LUT before batching, so the model sees stabilized intensities.
            self.vst_y_values = torch.from_numpy(lut.y_values)
            self.vst_f_values = torch.from_numpy(lut.f_values)

    def __len__(self):
        return len(self.samples)

    def get_sample_shape(self, idx):
        return self.samples.get_sample_shape(idx)

    def __getitem__(self, idx):
        input_tensor, target_tensor = self.samples[idx]

        if self.intensity_transform == "log1p":
            input_tensor = log1p_torch(input_tensor)
            target_tensor = log1p_torch(target_tensor)
        elif self.intensity_transform == "learned_vst":
            input_tensor = vst_forward_torch(input_tensor, self.vst_y_values, self.vst_f_values)
            target_tensor = vst_forward_torch(target_tensor, self.vst_y_values, self.vst_f_values)
        else:
            lam = _sample_lambda(
                fixed_lambda=self.boxcox_lam,
                lambda_conditioned=self.lambda_conditioned,
                lambda_min=self.lambda_min,
                lambda_max=self.lambda_max,
                lambda_candidates=self.lambda_candidates,
            )
            input_tensor = boxcox_1p_torch(input_tensor, lam=lam, eps=self.boxcox_eps)
            target_tensor = boxcox_1p_torch(target_tensor, lam=lam, eps=self.boxcox_eps)

            if self.lambda_conditioned:
                condition = lambda_condition_value(lam, lambda_min=self.lambda_min, lambda_max=self.lambda_max)
                lambda_map = torch.full_like(input_tensor, fill_value=condition)
                input_tensor = torch.cat((input_tensor, lambda_map), dim=0)

        return input_tensor, target_tensor


class NumpyFolderDataset(data.Dataset):
    """
    Historical name kept for compatibility.
    The dataset now supports both .npy and .lbf files.
    """

    def __init__(self, folder_path: str, crop_size: int | None = None):
        self.folder_path = folder_path
        self.crop_size = _normalize_crop_size(crop_size)
        self.files = _list_supported_files(folder_path)
        self._folder_shape = None

    def __len__(self):
        return len(self.files)

    def get_sample_shape(self, idx):
        if self.crop_size is not None:
            return self.crop_size, self.crop_size

        # Files inside one sequence folder are expected to share the same shape,
        # so one header read per folder is enough for batch grouping.
        if self._folder_shape is None:
            path = os.path.join(self.folder_path, self.files[0])
            self._folder_shape = _load_array_shape(path)
        return self._folder_shape

    def __getitem__(self, idx):
        try:
            path = os.path.join(self.folder_path, self.files[idx])
            img = _load_array(path)
            if self.crop_size is not None:
                img = center_crop(img, self.crop_size)
            img = np.ascontiguousarray(img)
            return torch.from_numpy(img).float().unsqueeze(0)
        except Exception as exc:
            print(f"Error loading {self.files[idx]}: {exc}")
            return self.__getitem__(random.randint(0, len(self) - 1))


class RandomPairDataset(data.Dataset):
    def __init__(self, dataset, intervals):
        self.dataset = dataset
        self.intervals = intervals

    def __len__(self):
        return len(self.dataset)

    def get_sample_shape(self, idx):
        return self.dataset.get_sample_shape(idx)

    def __getitem__(self, idx):
        interval = random.choice(self.intervals)
        lr = self.dataset[idx]

        if idx + interval < len(self.dataset):
            hr = self.dataset[idx + interval]
        elif idx - interval >= 0:
            hr = self.dataset[idx - interval]
        else:
            hr = self.dataset[idx]

        if random.random() < 0.5:
            return lr, hr
        return hr, lr


class ShapeConcatDataset(data.ConcatDataset):
    def get_sample_shape(self, idx):
        dataset_idx = 0
        sample_idx = idx
        for cumulative_size in self.cumulative_sizes:
            if idx < cumulative_size:
                break
            dataset_idx += 1
            sample_idx = idx - cumulative_size

        if dataset_idx > 0:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx].get_sample_shape(sample_idx)


def _parse_level_from_name(name: str):
    match = re.match(r"^\d+x\d+x(\d+)$", name)
    if match is None:
        return None
    return int(match.group(1))


def _parse_index_from_name(name: str) -> int | None:
    if re.fullmatch(r"\d+", name) is None:
        return None
    return int(name)


def _index_is_allowed(name: str, index_min: int | None, index_max: int | None) -> bool:
    if index_min is None and index_max is None:
        return True

    index = _parse_index_from_name(name)
    if index is None:
        return False
    if index_min is not None and index < index_min:
        return False
    if index_max is not None and index > index_max:
        return False
    return True


class CrossLevelIntervalPairDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        intervals,
        crop_size=512,
        npy_folder_name="npy",
        strict_data_subdir: bool = False,
        data_index_min: int | None = None,
        data_index_max: int | None = None,
        include_levels: tuple[int, ...] | None = None,
    ):
        self.root_dir = root_dir
        self.intervals = intervals
        self.crop_size = crop_size
        self.preferred_subdirs = _normalize_preferred_subdirs(
            npy_folder_name,
            include_defaults=not strict_data_subdir,
        )
        self.strict_data_subdir = strict_data_subdir
        self.data_index_min = data_index_min
        self.data_index_max = data_index_max
        # 只保留指定叠加层级（如 level 2/3/4），把 level1 留作 OOD 测试。
        # 跨层级配对的 target 也只会落在保留的层级上（candidate_levels 由 group_datasets 决定）。
        self.include_levels = set(include_levels) if include_levels else None

        self.group_datasets = {}
        self.input_items = []
        self.levels = set()
        skipped_missing_data_dirs = []

        for level_name in sorted(os.listdir(root_dir)):
            level_path = os.path.join(root_dir, level_name)
            if not os.path.isdir(level_path):
                continue

            level = _parse_level_from_name(level_name)
            if level is None:
                continue
            if self.include_levels is not None and level not in self.include_levels:
                continue

            for subdir in sorted(os.listdir(level_path)):
                if not _index_is_allowed(subdir, self.data_index_min, self.data_index_max):
                    continue

                subdir_path = os.path.join(level_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                data_dirs = _resolve_data_dirs(
                    subdir_path,
                    self.preferred_subdirs,
                    strict_subdir=self.strict_data_subdir,
                )
                if len(data_dirs) == 0:
                    if self.strict_data_subdir:
                        skipped_missing_data_dirs.append(subdir_path)
                    continue

                for data_dir in data_dirs:
                    dataset = NumpyFolderDataset(data_dir, crop_size=self.crop_size)
                    if len(dataset) == 0:
                        continue

                    data_name = os.path.relpath(data_dir, subdir_path)
                    key = (level, subdir, data_name)
                    self.group_datasets[key] = dataset
                    self.levels.add(level)

                    for frame_idx in range(len(dataset)):
                        self.input_items.append((level, subdir, data_name, frame_idx))

        self.levels = sorted(self.levels)

        if skipped_missing_data_dirs:
            preview = ", ".join(skipped_missing_data_dirs[:10])
            if len(skipped_missing_data_dirs) > 10:
                preview += f", ... (+{len(skipped_missing_data_dirs) - 10} more)"
            print(
                f"[INFO] Skipped {len(skipped_missing_data_dirs)} directories without "
                f"requested data subdir {self.preferred_subdirs[0]!r}: {preview}"
            )

        if len(self.input_items) == 0:
            raise RuntimeError(
                f"No supported training files ({', '.join(SUPPORTED_ARRAY_EXTS)}) were found under {root_dir}"
            )

    def __len__(self):
        return len(self.input_items)

    def get_sample_shape(self, idx):
        input_level, subdir, data_name, input_idx = self.input_items[idx]
        return self.group_datasets[(input_level, subdir, data_name)].get_sample_shape(input_idx)

    @staticmethod
    def _pair_index(idx, interval, length):
        if idx + interval < length:
            return idx + interval
        if idx - interval >= 0:
            return idx - interval
        if idx < length:
            return idx
        return length - 1

    def __getitem__(self, idx):
        input_level, subdir, data_name, input_idx = self.input_items[idx]
        input_dataset = self.group_datasets[(input_level, subdir, data_name)]
        input_tensor = input_dataset[input_idx]

        candidate_levels = [
            level
            for level in self.levels
            if level >= input_level and (level, subdir, data_name) in self.group_datasets
        ]
        if len(candidate_levels) == 0:
            candidate_levels = [input_level]

        target_level = random.choice(candidate_levels)
        target_dataset = self.group_datasets[(target_level, subdir, data_name)]

        interval = random.choice(self.intervals)
        target_idx = self._pair_index(input_idx, interval, len(target_dataset))
        target_tensor = target_dataset[target_idx]

        return input_tensor, target_tensor


def create_train_dataset(
    train_dir,
    intervals,
    crop_size=512,
    npy_folder_name="npy",
    strict_data_subdir: bool = False,
    data_index_min: int | None = None,
    data_index_max: int | None = None,
    include_levels: tuple[int, ...] | None = None,
):
    preferred_subdirs = _normalize_preferred_subdirs(
        npy_folder_name,
        include_defaults=not strict_data_subdir,
    )

    level_dirs = [
        subdir
        for subdir in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, subdir)) and _parse_level_from_name(subdir) is not None
    ]

    if len(level_dirs) > 0:
        return CrossLevelIntervalPairDataset(
            train_dir,
            intervals,
            crop_size=crop_size,
            npy_folder_name=preferred_subdirs,
            strict_data_subdir=strict_data_subdir,
            data_index_min=data_index_min,
            data_index_max=data_index_max,
            include_levels=include_levels,
        )

    datasets = []
    skipped_missing_data_dirs = []

    direct_data_dir = _resolve_data_dir(train_dir, preferred_subdirs, strict_subdir=strict_data_subdir)
    if direct_data_dir is not None and direct_data_dir == train_dir:
        base_dataset = NumpyFolderDataset(direct_data_dir, crop_size=crop_size)
        if len(base_dataset) > 0:
            datasets.append(RandomPairDataset(base_dataset, intervals))

    for subdir in sorted(os.listdir(train_dir)):
        if not _index_is_allowed(subdir, data_index_min, data_index_max):
            continue

        subdir_path = os.path.join(train_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        data_dirs = _resolve_data_dirs(subdir_path, preferred_subdirs, strict_subdir=strict_data_subdir)
        if len(data_dirs) == 0:
            if strict_data_subdir:
                skipped_missing_data_dirs.append(subdir_path)
            continue

        for data_dir in data_dirs:
            dataset = NumpyFolderDataset(data_dir, crop_size=crop_size)
            if len(dataset) == 0:
                continue

            # Each RandomPairDataset is scoped to one concrete folder such as
            # mix/0/npy or mix/0/lbf, so frame pairing never crosses sequences
            # or mixes npy samples with lbf samples.
            datasets.append(RandomPairDataset(dataset, intervals))

    if skipped_missing_data_dirs:
        preview = ", ".join(skipped_missing_data_dirs[:10])
        if len(skipped_missing_data_dirs) > 10:
            preview += f", ... (+{len(skipped_missing_data_dirs) - 10} more)"
        print(
            f"[INFO] Skipped {len(skipped_missing_data_dirs)} directories without "
            f"requested data subdir {preferred_subdirs[0]!r}: {preview}"
        )

    if len(datasets) == 0:
        raise RuntimeError(
            f"No supported training files ({', '.join(SUPPORTED_ARRAY_EXTS)}) were found under {train_dir}"
        )

    if len(datasets) == 1:
        return datasets[0]

    return ShapeConcatDataset(datasets)
