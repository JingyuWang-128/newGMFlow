import random
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Sampler


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_NAMES: Tuple[str, ...] = ("COCO", "CelebA-HQ", "DIV2K", "Paris_StreetView")


def _collect_images(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _resolve_roots(data_path: str, dataset_names: Sequence[str]) -> List[Path]:
    base = Path(data_path).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"data_path does not exist: {base}")

    sub_roots = [base / name for name in dataset_names if (base / name).is_dir()]
    if sub_roots:
        return sub_roots
    return [base]


def _interleave_paths_with_ids(
    per_root_paths: Sequence[List[Path]],
) -> Tuple[List[Path], List[int]]:
    """Round-robin 交替合并多数据集路径，并记录每条路径的 dataset_id。"""
    mixed_paths: List[Path] = []
    mixed_ids: List[int] = []
    depth = 0
    while True:
        added = False
        for ds_id, paths in enumerate(per_root_paths):
            if depth < len(paths):
                mixed_paths.append(paths[depth])
                mixed_ids.append(ds_id)
                added = True
        if not added:
            break
        depth += 1
    return mixed_paths, mixed_ids


def _build_mixed_paths(
    data_path: str,
    seed: int = 42,
    dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES,
) -> Tuple[List[Path], List[int]]:
    """返回 (paths, dataset_ids)，保证路径与 id 一一对应。"""
    roots = _resolve_roots(data_path, dataset_names)
    rng = random.Random(seed)
    per_root_paths: List[List[Path]] = []
    for root in roots:
        paths = _collect_images(root)
        rng.shuffle(paths)
        if paths:
            per_root_paths.append(paths)

    if not per_root_paths:
        raise RuntimeError(f"No images found under: {data_path}")
    return _interleave_paths_with_ids(per_root_paths)


class CelebADataset(Dataset):
    """
    多数据集混合 Dataset。
    - 若 data_path 下有 COCO/CelebA-HQ/DIV2K/Paris_StreetView，则按 round-robin 混合。
    - 否则递归读取 data_path 下所有图片。
    - 通过 dataset_ids 支持分层采样，保证每个 batch 来自不同数据集。
    """

    def __init__(
        self,
        data_path: str,
        transform=None,
        split: str = "train",
        test_ratio: float = 0.1,
        split_seed: int = 42,
        dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES,
    ):
        if split not in {"train", "test", "all"}:
            raise ValueError(f"split must be train/test/all, got: {split}")
        if not (0.0 < test_ratio < 1.0):
            raise ValueError(f"test_ratio must be in (0,1), got: {test_ratio}")

        self.transform = transform
        all_paths, all_dataset_ids = _build_mixed_paths(
            data_path=data_path,
            seed=split_seed,
            dataset_names=dataset_names,
        )

        split_idx = int(len(all_paths) * (1.0 - test_ratio))
        split_idx = max(1, min(split_idx, len(all_paths) - 1))

        if split == "train":
            self.paths = all_paths[:split_idx]
            self.dataset_ids = all_dataset_ids[:split_idx]
        elif split == "test":
            self.paths = all_paths[split_idx:]
            self.dataset_ids = all_dataset_ids[split_idx:]
        else:
            self.paths = all_paths
            self.dataset_ids = all_dataset_ids

        self.labels = [0] * len(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


class StratifiedDistributedBatchSampler(Sampler[List[int]]):
    """
    分层 BatchSampler：结合 DistributedSampler 与分层采样，
    保证每个 batch 中样本来自多个数据集，同一 batch 内不出现全部来自同一数据集的情况。
    """

    def __init__(
        self,
        dataset: CelebADataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # 按 dataset_id 分组索引
        n = len(dataset)
        self.dataset_ids = dataset.dataset_ids
        self.indices_by_dataset: List[List[int]] = []
        num_datasets = max(self.dataset_ids) + 1
        for ds_id in range(num_datasets):
            self.indices_by_dataset.append(
                [i for i in range(n) if self.dataset_ids[i] == ds_id]
            )

        # 本 rank 的样本量（与 DistributedSampler 一致的划分）
        self.num_samples = (n + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas

    def _get_shard_indices_by_dataset(self) -> List[List[int]]:
        """按 DDP 划分 + shuffle 后，得到本 rank 的分层索引。"""
        rng = random.Random(self.seed + self.epoch)
        n = len(self.dataset)

        # 1. 在每个 dataset 内先 shuffle
        shuffled_by_ds: List[List[int]] = []
        for ids in self.indices_by_dataset:
            lst = list(ids)
            rng.shuffle(lst)
            shuffled_by_ds.append(lst)

        # 2. 交替采样形成混合序列（保证相邻样本来自不同数据集）
        mixed: List[int] = []
        max_len = max(len(lst) for lst in shuffled_by_ds)
        for depth in range(max_len):
            for ds_id, lst in enumerate(shuffled_by_ds):
                if depth < len(lst):
                    mixed.append(lst[depth])

        # 3.  Padding 以整除 num_replicas（允许重复抽样）
        if len(mixed) < self.total_size:
            need = self.total_size - len(mixed)
            padding = list(rng.choices(mixed, k=need))
            mixed = mixed + padding
        else:
            mixed = mixed[: self.total_size]

        # 4. 分配到各 rank
        indices = mixed[self.rank : self.total_size : self.num_replicas]
        return indices

    def __iter__(self):
        indices = self._get_shard_indices_by_dataset()
        n = len(indices)
        if self.drop_last and n % self.batch_size != 0:
            n = (n // self.batch_size) * self.batch_size
        for i in range(0, n, self.batch_size):
            yield indices[i : i + self.batch_size]

    def __len__(self) -> int:
        n = self.num_samples
        if self.drop_last and n % self.batch_size != 0:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch
