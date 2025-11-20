"""
CAP6415 F25 — Project #8
Data utilities: CIFAR-10 datasets, transforms, and DataLoaders.

Purpose:
    Builds train/val (and optional test) DataLoaders for CIFAR-10 with standard
    augmentations. When fine-tuning, can oversample a specified focus class to
    emphasize harder categories.

Primary API:
    get_dataloaders(cfg: dict, eval_mode: bool = False)
        Builds DataLoaders according to the YAML config.

Key config keys:
    dataset.data_dir (str): Cache/download location for CIFAR-10.
    train.batch_size (int), train.num_workers (int), train.pin_memory (bool)
    target.focus_class (str|None): Class name to oversample (e.g., "cat").
    target.oversample (bool), target.oversample_factor (int): Rebalancing strength.

Returns:
    tuple:
        train_loader (DataLoader or None): None when eval_mode=True.
        val_loader   (DataLoader): Validation split.
        test_loader  (DataLoader or None): If implemented; else None.
        class_names  (List[str]): CIFAR-10 class names in loader order.

Example:
    train_loader, val_loader, _, class_names = get_dataloaders(cfg, eval_mode=False)
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# CIFAR-10 stats
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

def _build_transforms(eval_mode: bool = False):
    if eval_mode:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    # stronger but classic CIFAR-10 stack
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])

def _make_sampler_if_needed(train_set, class_names, cfg):
    tgt_cfg = cfg.get("target", {})
    if not tgt_cfg.get("oversample", False):
        return None

    focus_name = tgt_cfg.get("focus_class")
    assert focus_name in class_names, f"Unknown class '{focus_name}'"
    focus_idx = class_names.index(focus_name)

    targets = torch.as_tensor(train_set.targets)
    weights = torch.ones_like(targets, dtype=torch.float)
    factor = float(tgt_cfg.get("oversample_factor", 4.0))
    weights[targets == focus_idx] = factor

    # sample with replacement so epochs stay the same length
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def get_dataloaders(cfg, eval_mode: bool = False):
    root = cfg["dataset"]["data_dir"]
    nw = int(cfg["dataset"].get("num_workers", 2))
    bs = int(cfg["train"]["batch_size"])

    t_train = _build_transforms(eval_mode=False)
    t_eval  = _build_transforms(eval_mode=True)

    train_set = None
    sampler = None
    if not eval_mode:
        train_set = datasets.CIFAR10(root=root, train=True, transform=t_train, download=True)
        class_names = list(train_set.classes)
        sampler = _make_sampler_if_needed(train_set, class_names, cfg)
    else:
        class_names = list(CIFAR10_CLASSES)

    val_set = datasets.CIFAR10(root=root, train=False, transform=t_eval, download=True)
    # keep class_names consistent with loader order
    class_names = list(val_set.classes)

    train_loader = None
    if train_set is not None:
        train_loader = DataLoader(
            train_set, batch_size=bs, shuffle=(sampler is None),
            sampler=sampler, num_workers=nw, pin_memory=True
        )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True
    )
    return train_loader, val_loader, None, class_names
