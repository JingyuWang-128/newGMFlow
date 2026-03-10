import random
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


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


def _interleave_paths(per_root_paths: Sequence[List[Path]]) -> List[Path]:
    mixed: List[Path] = []
    depth = 0
    while True:
        added = False
        for paths in per_root_paths:
            if depth < len(paths):
                mixed.append(paths[depth])
                added = True
        if not added:
            break
        depth += 1
    return mixed


def _build_mixed_paths(
    data_path: str,
    seed: int = 42,
    dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES,
) -> List[Path]:
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
    return _interleave_paths(per_root_paths)


class CelebADataset(Dataset):
    """
    Mixed image dataset.
    - If `data_path` contains COCO/CelebA-HQ/DIV2K/Paris_StreetView, it interleaves all four.
    - Otherwise it reads images recursively from `data_path`.
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
        all_paths = _build_mixed_paths(
            data_path=data_path,
            seed=split_seed,
            dataset_names=dataset_names,
        )

        split_idx = int(len(all_paths) * (1.0 - test_ratio))
        split_idx = max(1, min(split_idx, len(all_paths) - 1))

        if split == "train":
            self.paths = all_paths[:split_idx]
        elif split == "test":
            self.paths = all_paths[split_idx:]
        else:
            self.paths = all_paths

        # Keep compatibility with code that expects tuple(image, label).
        self.labels = [0] * len(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]
