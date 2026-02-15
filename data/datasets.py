"""
GenMamba-Flow 数据集与 DataLoader
从指定目录递归收集图片，按比例划分训练/验证/测试集，供训练与测试使用。
"""

import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def collect_image_paths(root: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    """递归收集目录下所有图片路径。"""
    root = Path(root)
    if not root.exists():
        return []
    paths = []
    for ext in extensions:
        paths.extend(root.rglob(f"*{ext}"))
    return [str(p) for p in sorted(paths)]


def split_paths(paths: List[str], ratios: List[float], seed: int = 42) -> List[List[str]]:
    """将路径列表按比例划分为多份（如 train/val/test）。ratios 如 [0.8, 0.1, 0.1]，和应为 1.0。"""
    if not paths:
        return [[] for _ in ratios]
    rng = random.Random(seed)
    indices = list(range(len(paths)))
    rng.shuffle(indices)
    n = len(indices)
    out = []
    start = 0
    for i, r in enumerate(ratios):
        end = start + int(round(n * r))
        if i == len(ratios) - 1:
            end = n
        out.append([paths[j] for j in indices[start:end]])
        start = end
    return out


def get_cover_and_secret_paths_from_config(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """从 config 的 cover_roots / secret_roots 收集全部路径。兼容旧配置 train_datasets/secret_datasets。"""
    data_cfg = config.get("data", {})
    cover_roots = data_cfg.get("cover_roots", [])
    secret_roots = data_cfg.get("secret_roots", [])
    if not cover_roots and data_cfg.get("train_datasets"):
        cover_roots = [ds.get("root", "") for ds in data_cfg["train_datasets"]]
    if not secret_roots and data_cfg.get("secret_datasets"):
        secret_roots = [s.get("root", "") for s in data_cfg["secret_datasets"]]
    if not cover_roots:
        cover_roots = ["./data/placeholder/DIV2K"]
    if not secret_roots:
        secret_roots = ["./data/placeholder/Paris_StreetView"]
    all_cover = []
    for r in cover_roots:
        all_cover.extend(collect_image_paths(r))
    all_secret = []
    for r in secret_roots:
        all_secret.extend(collect_image_paths(r))
    return all_cover, all_secret


def get_split_paths_from_config(config: Dict[str, Any]) -> Tuple[
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str]],
]:
    """根据 config 收集路径并按 split_ratios 划分。返回 (train_cover, train_secret), (val_cover, val_secret), (test_cover, test_secret)。"""
    all_cover, all_secret = get_cover_and_secret_paths_from_config(config)
    ratios = config.get("data", {}).get("split_ratios", [0.8, 0.1, 0.1])
    seed = config.get("data", {}).get("split_seed", 42)
    if len(ratios) != 3:
        ratios = [0.8, 0.1, 0.1]
    train_cover, val_cover, test_cover = split_paths(all_cover, ratios, seed)
    train_secret, val_secret, test_secret = split_paths(all_secret, ratios, seed)
    return (train_cover, train_secret), (val_cover, val_secret), (test_cover, test_secret)


def print_split_stats(config: Dict[str, Any]) -> None:
    """打印训练/验证/测试集划分数量。"""
    (tc, ts), (vc, vs), (xc, xs) = get_split_paths_from_config(config)
    print("[Data split] cover: train=%d val=%d test=%d | secret: train=%d val=%d test=%d" % (len(tc), len(vc), len(xc), len(ts), len(vs), len(xs)))


class ImagePathsDataset(Dataset):
    """从路径列表加载图像。"""

    def __init__(self, paths: List[str], image_size: int = 256, is_cover: bool = True):
        self.paths = paths
        self.image_size = image_size
        self.is_cover = is_cover
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip() if is_cover else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths) if self.paths else 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.paths:
            key = "image" if self.is_cover else "secret"
            return {key: torch.zeros(3, self.image_size, self.image_size)}
        img = Image.open(self.paths[idx % len(self.paths)]).convert("RGB")
        key = "image" if self.is_cover else "secret"
        return {key: self.transform(img)}


class StegoPairPathDataset(Dataset):
    """隐写对：从覆盖图路径列表与秘密图路径列表组成 (cover, secret, text) 对。"""

    def __init__(
        self,
        cover_paths: List[str],
        secret_paths: List[str],
        image_size: int = 256,
        secret_size: int = 256,
        max_samples: Optional[int] = None,
        text_prompts: Optional[List[str]] = None,
    ):
        self.cover_paths = cover_paths
        self.secret_paths = secret_paths
        self.image_size = image_size
        self.secret_size = secret_size
        self.max_samples = max_samples or (len(cover_paths) * max(1, len(secret_paths)))
        self.text_prompts = text_prompts or ["a natural image"]
        self.cover_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.secret_transform = transforms.Compose([
            transforms.Resize((secret_size, secret_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return min(self.max_samples, max(1, len(self.cover_paths)) * max(1, len(self.secret_paths)))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ncover = max(1, len(self.cover_paths))
        nsecret = max(1, len(self.secret_paths))
        cidx = idx % ncover
        sidx = (idx // ncover) % nsecret
        cover_img = Image.open(self.cover_paths[cidx]).convert("RGB") if self.cover_paths else Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        secret_img = Image.open(self.secret_paths[sidx]).convert("RGB") if self.secret_paths else Image.new("RGB", (self.secret_size, self.secret_size), (128, 128, 128))
        cover = self.cover_transform(cover_img)
        secret = self.secret_transform(secret_img)
        text = self.text_prompts[idx % len(self.text_prompts)]
        return {"cover": cover, "secret": secret, "text": text}


def _make_stego_dataloader(
    config: Dict[str, Any],
    cover_paths: List[str],
    secret_paths: List[str],
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    data_cfg = config.get("data", {})
    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)
    img_size = data_cfg.get("image_size", 256)
    secret_size = data_cfg.get("secret_size", 256)
    max_samples = data_cfg.get("num_train_samples")
    dataset = StegoPairPathDataset(cover_paths, secret_paths, img_size, secret_size, max_samples=max_samples)
    n = len(dataset)
    if n == 0:
        raise ValueError("训练集样本数为 0。请检查 data.cover_roots 与 data.secret_roots 下是否有图片，或调整 data.split_ratios。")
    min_required = batch_size * world_size if world_size > 1 else batch_size
    drop_last = drop_last and n >= min_required
    if world_size > 1 and n < min_required and rank == 0:
        import warnings
        warnings.warn("训练集样本数 %d < batch_size*world_size (%d)，已关闭 drop_last。建议增加数据或减少 GPU 数量/减小 batch_size。" % (n, min_required))
    sampler = None
    if world_size > 1 and shuffle:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last,
    )


def get_train_dataloader(config: Dict[str, Any], rank: int = 0, world_size: int = 1, shuffle: bool = True) -> DataLoader:
    """获取训练集 DataLoader（使用划分后的训练集）。支持 DDP。"""
    (train_cover, train_secret), _, _ = get_split_paths_from_config(config)
    if not train_cover or not train_secret:
        data_cfg = config.get("data", {})
        cover_roots = data_cfg.get("cover_roots", ["./data/placeholder/DIV2K"])
        secret_roots = data_cfg.get("secret_roots", ["./data/placeholder/Paris_StreetView"])
        train_cover = []
        for r in cover_roots:
            train_cover.extend(collect_image_paths(r))
        train_secret = []
        for r in secret_roots:
            train_secret.extend(collect_image_paths(r))
    return _make_stego_dataloader(config, train_cover, train_secret, rank=rank, world_size=world_size, shuffle=shuffle, drop_last=True)


def get_val_dataloader(config: Dict[str, Any], batch_size: Optional[int] = None) -> DataLoader:
    """获取验证集 DataLoader。"""
    _, (val_cover, val_secret), _ = get_split_paths_from_config(config)
    if not val_cover or not val_secret:
        return get_train_dataloader(config, rank=0, world_size=1)
    data_cfg = config.get("data", {})
    bs = batch_size or data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)
    img_size = data_cfg.get("image_size", 256)
    secret_size = data_cfg.get("secret_size", 256)
    dataset = StegoPairPathDataset(val_cover, val_secret, img_size, secret_size, max_samples=None)
    return DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers, drop_last=False)


def get_test_dataloader(config: Dict[str, Any], batch_size: Optional[int] = None) -> DataLoader:
    """获取测试集 DataLoader。"""
    _, _, (test_cover, test_secret) = get_split_paths_from_config(config)
    if not test_cover or not test_secret:
        data_cfg = config.get("data", {})
        cover_roots = data_cfg.get("cover_roots", ["./data/placeholder/DIV2K"])
        secret_roots = data_cfg.get("secret_roots", ["./data/placeholder/Paris_StreetView"])
        test_cover = []
        for r in cover_roots:
            test_cover.extend(collect_image_paths(r))
        test_secret = []
        for r in secret_roots:
            test_secret.extend(collect_image_paths(r))
    data_cfg = config.get("data", {})
    bs = batch_size or data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)
    img_size = data_cfg.get("image_size", 256)
    secret_size = data_cfg.get("secret_size", 256)
    dataset = StegoPairPathDataset(test_cover, test_secret, img_size, secret_size, max_samples=None)
    return DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers, drop_last=False)


def get_secret_train_dataloader(config: Dict[str, Any], batch_size: Optional[int] = None) -> DataLoader:
    """获取仅秘密图像训练集 DataLoader（用于 RQ-VAE 阶段1）。"""
    (_, train_secret), _, _ = get_split_paths_from_config(config)
    if not train_secret:
        secret_roots = config.get("data", {}).get("secret_roots", ["./data/placeholder/Paris_StreetView"])
        train_secret = []
        for r in secret_roots:
            train_secret.extend(collect_image_paths(r))
    data_cfg = config.get("data", {})
    bs = batch_size or data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)
    secret_size = data_cfg.get("secret_size", 256)
    dataset = ImagePathsDataset(train_secret, secret_size, is_cover=False)
    return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)


def get_secret_dataloader(config: Dict[str, Any], batch_size: Optional[int] = None) -> DataLoader:
    """获取秘密图像 DataLoader（全量）。"""
    data_cfg = config.get("data", {})
    bs = batch_size or data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)
    secret_roots = data_cfg.get("secret_roots", data_cfg.get("secret_datasets", [{}]))
    if isinstance(secret_roots, list) and secret_roots and isinstance(secret_roots[0], dict):
        secret_roots = [s.get("root", "") for s in secret_roots]
    if not secret_roots:
        secret_roots = ["./data/placeholder/Paris_StreetView"]
    all_secret = []
    for r in secret_roots:
        all_secret.extend(collect_image_paths(r))
    dataset = ImagePathsDataset(all_secret, data_cfg.get("secret_size", 256), is_cover=False)
    return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)


def build_dataset_from_config(config: Dict[str, Any]) -> tuple:
    """从 config 构建训练用覆盖与秘密数据集。"""
    (train_cover, train_secret), _, _ = get_split_paths_from_config(config)
    data_cfg = config.get("data", {})
    img_size = data_cfg.get("image_size", 256)
    secret_size = data_cfg.get("secret_size", 256)
    cover_ds = ImagePathsDataset(train_cover, img_size, is_cover=True) if train_cover else ImagePathsDataset([], img_size, is_cover=True)
    secret_ds = ImagePathsDataset(train_secret, secret_size, is_cover=False) if train_secret else ImagePathsDataset([], secret_size, is_cover=False)
    return cover_ds, secret_ds
