"""
data.py — CIFAR-10 dataloaders with optional class-focused oversampling
and mild class-conditional augmentations for the focus class.
"""
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms

CIFAR10_CLASSES: List[str] = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def _normalize():
    return transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))

class _FocusAugment(Dataset):
    """
    Wrap a dataset and, if label == focus_idx, apply extra mild augs to
    help robustness on the targeted class.
    """
    def __init__(self, base: Dataset, focus_idx: int | None):
        self.base = base
        self.focus_idx = focus_idx
        self.extra = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3))
        ])

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        if self.focus_idx is not None and y == self.focus_idx:
            img = self.extra(img)
        return img, y

def _transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize(),
        ])
    else:
        return transforms.Compose([transforms.ToTensor(), _normalize()])

def get_dataloaders(cfg, eval_mode: bool=False) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    root = cfg["dataset"]["data_dir"]
    num_workers = int(cfg["dataset"].get("num_workers", 2))
    bs = int(cfg["train"]["batch_size"])
    pin = torch.cuda.is_available()  # avoid CPU pin_memory warning

    train_full = datasets.CIFAR10(root, train=True, download=True, transform=_transforms(train=True))
    test_set   = datasets.CIFAR10(root, train=False, download=True, transform=_transforms(train=False))

    val_size   = 5000
    train_size = len(train_full) - val_size
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=gen)

    sampler = None
    focus_idx = None
    if (not eval_mode) and cfg["target"].get("oversample") and cfg["target"].get("focus_class") is not None:
        fc = cfg["target"]["focus_class"]
        focus_idx = CIFAR10_CLASSES.index(fc) if isinstance(fc, str) else int(fc)
        factor = float(cfg["target"].get("oversample_factor", 3.0))
        labels = torch.tensor([train_ds.dataset.targets[i] for i in train_ds.indices])
        weights = torch.ones(len(labels))
        weights[labels == focus_idx] = factor
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Apply extra augs only on training subset
    if focus_idx is not None and not eval_mode:
        train_ds = _FocusAugment(train_ds, focus_idx)
    else:
        train_ds = _FocusAugment(train_ds, None)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_set, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader, test_loader, CIFAR10_CLASSES
