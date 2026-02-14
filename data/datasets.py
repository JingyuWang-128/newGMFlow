"""
GenMamba-Flow 数据集与 DataLoader
支持 DIV2K / COCO / 通用图像目录，以及秘密图像（Paris StreetView, CelebA-HQ 等）
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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


class ImageFolderDataset(Dataset):
    """通用图像目录数据集。"""

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        max_samples: Optional[int] = None,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.paths = collect_image_paths(str(self.root), extensions)
        if max_samples is not None and len(self.paths) > max_samples:
            self.paths = random.Random(42).sample(self.paths, max_samples)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths) if self.paths else 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.paths:
            return {"image": torch.zeros(3, self.image_size, self.image_size)}
        path = self.paths[idx % len(self.paths)]
        img = Image.open(path).convert("RGB")
        return {"image": self.transform(img)}


class DIV2KLikeDataset(ImageFolderDataset):
    """DIV2K 风格：root 下可有 train/val 子目录或直接为图片。"""

    def __init__(self, root: str, split: str = "train", image_size: int = 256, max_samples: Optional[int] = None):
        sub = Path(root) / split if split else Path(root)
        super().__init__(str(sub), image_size=image_size, max_samples=max_samples)


class SecretImageDataset(Dataset):
    """秘密图像数据集，用于 RQ-VAE 编码与解码训练/测试。"""

    def __init__(self, root: str, image_size: int = 256, max_samples: Optional[int] = None):
        self.paths = collect_image_paths(root)
        self.image_size = image_size
        if max_samples is not None and len(self.paths) > max_samples:
            self.paths = random.Random(43).sample(self.paths, max_samples)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths) if self.paths else 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.paths:
            return {"secret": torch.zeros(3, self.image_size, self.image_size)}
        img = Image.open(self.paths[idx % len(self.paths)]).convert("RGB")
        return {"secret": self.transform(img)}


class StegoPairDataset(Dataset):
    """
    隐写对数据集：覆盖图像（用于生成条件/文本） + 秘密图像。
    若提供 text_prompts 列表则使用，否则用占位文本。
    """

    def __init__(
        self,
        cover_roots: List[str],
        secret_root: str,
        image_size: int = 256,
        secret_size: int = 256,
        max_samples: Optional[int] = None,
        text_prompts: Optional[List[str]] = None,
    ):
        self.cover_paths = []
        for r in cover_roots:
            self.cover_paths.extend(collect_image_paths(r))
        self.secret_dataset = SecretImageDataset(secret_root, image_size=secret_size, max_samples=None)
        self.image_size = image_size
        self.secret_size = secret_size
        self.max_samples = max_samples or (len(self.cover_paths) * max(1, len(self.secret_dataset)))
        self.text_prompts = text_prompts or ["a natural image"]
        self.cover_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return min(self.max_samples, max(1, len(self.cover_paths)) * max(1, len(self.secret_dataset)))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cidx = idx % len(self.cover_paths) if self.cover_paths else 0
        sidx = (idx // max(1, len(self.cover_paths))) % len(self.secret_dataset)
        cover_img = Image.open(self.cover_paths[cidx]).convert("RGB") if self.cover_paths else Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        cover = self.cover_transform(cover_img)
        secret_item = self.secret_dataset[sidx]
        secret = secret_item["secret"]
        text = self.text_prompts[idx % len(self.text_prompts)]
        return {
            "cover": cover,
            "secret": secret,
            "text": text,
        }


def build_dataset_from_config(config: Dict[str, Any]) -> tuple:
    """
    从 config 构建训练用覆盖数据集与秘密数据集。
    Returns:
        cover_dataset, secret_dataset
    """
    data_cfg = config.get("data", {})
    img_size = data_cfg.get("image_size", 256)
    secret_size = data_cfg.get("secret_size", 256)
    max_samples = data_cfg.get("num_train_samples")

    cover_datasets = []
    for ds in data_cfg.get("train_datasets", []):
        root = ds.get("root", "")
        split = ds.get("split", "train")
        d = DIV2KLikeDataset(root, split=split, image_size=img_size, max_samples=max_samples)
        if len(d) > 0:
            cover_datasets.append(d)
    if not cover_datasets:
        cover_datasets = [ImageFolderDataset(data_cfg.get("train_datasets", [{}])[0].get("root", "./data/placeholder/DIV2K"), image_size=img_size)]

    secret_roots = [s.get("root", "") for s in data_cfg.get("secret_datasets", [])]
    if not secret_roots:
        secret_roots = ["./data/placeholder/Paris_StreetView"]
    secret_dataset = ConcatDataset([
        SecretImageDataset(r, image_size=secret_size, max_samples=max_samples // len(secret_roots)) for r in secret_roots
    ])

    cover_dataset = ConcatDataset(cover_datasets)
    return cover_dataset, secret_dataset


def get_train_dataloader(
    config: Dict[str, Any],
    use_stego_pair: bool = True,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
) -> DataLoader:
    """获取训练 DataLoader。支持 DDP：传入 rank/world_size 时使用 DistributedSampler。"""
    data_cfg = config.get("data", {})
    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)
    img_size = data_cfg.get("image_size", 256)
    secret_size = data_cfg.get("secret_size", 256)
    cover_roots = []
    for ds in data_cfg.get("train_datasets", []):
        cover_roots.append(ds.get("root", ""))
    if not cover_roots:
        cover_roots = ["./data/placeholder/DIV2K"]
    secret_roots = [s.get("root", "") for s in data_cfg.get("secret_datasets", [])]
    if not secret_roots:
        secret_roots = ["./data/placeholder/Paris_StreetView"]
    dataset = StegoPairDataset(
        cover_roots=cover_roots,
        secret_root=secret_roots[0],
        image_size=img_size,
        secret_size=secret_size,
        max_samples=data_cfg.get("num_train_samples"),
    )
    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_secret_dataloader(config: Dict[str, Any], batch_size: Optional[int] = None) -> DataLoader:
    """获取秘密图像 DataLoader（用于 RQ-VAE 预训练或解码评估）。"""
    data_cfg = config.get("data", {})
    bs = batch_size or data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)
    secret_roots = [s.get("root", "") for s in data_cfg.get("secret_datasets", [])]
    if not secret_roots:
        secret_roots = ["./data/placeholder/Paris_StreetView"]
    dataset = ConcatDataset([
        SecretImageDataset(r, image_size=data_cfg.get("secret_size", 256)) for r in secret_roots
    ])
    return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)
