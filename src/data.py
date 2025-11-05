"""
data.py
───────
Data pipeline:
- CIFAR-10 dataset with standard train/val/test splits.
- Basic augmentations for training.
- Optional WeightedRandomSampler to oversample a focus class during fine-tuning.
"""

from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms

# CIFAR-10 class labels in order
CIFAR10_CLASSES: List[str] = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def _normalize():
    # Official CIFAR-10 normalization
    return transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))

def _transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize(),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            _normalize(),
        ])

def get_dataloaders(cfg, eval_mode: bool=False) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train/val/test dataloaders.
    If `eval_mode` is True, training oversampling is disabled.
    """
    root = cfg["dataset"]["data_dir"]
    num_workers = int(cfg["dataset"].get("num_workers", 2))
    bs = int(cfg["train"]["batch_size"])

    # Download datasets if missing
    train_full = datasets.CIFAR10(root, train=True,  download=True, transform=_transforms(train=True))
    test_set   = datasets.CIFAR10(root, train=False, download=True, transform=_transforms(train=False))

    # Validation split from the training set
    val_size   = 5000
    train_size = len(train_full) - val_size
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=gen)

    # Optional: oversample the focus class during training
    sampler = None
    if (not eval_mode) and cfg["target"].get("oversample") and cfg["target"].get("focus_class") is not None:
        focus = cfg["target"]["focus_class"]
        focus_idx = CIFAR10_CLASSES.index(focus) if isinstance(focus, str) else int(focus)
        factor = float(cfg["target"].get("oversample_factor", 3.0))

        # Build label vector for the train subset using original dataset targets
        labels = torch.tensor([train_ds.dataset.targets[i] for i in train_ds.indices])
        weights = torch.ones(len(labels))
        weights[labels == focus_idx] = factor
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, CIFAR10_CLASSES
