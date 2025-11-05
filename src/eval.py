# src/eval.py
# Evaluate a trained checkpoint on CIFAR-10 and save:
#  - per-class precision table (printed)
#  - overall accuracy (printed)
#  - confusion matrix image
#  - sample grids of correct/incorrect predictions

import argparse
import os
from pathlib import Path
from typing import Tuple, List

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as tvu

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# CIFAR-10 class order used by torchvision
CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ------------------------- utilities ------------------------- #
def set_seed(seed: int) -> None:
    import random
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True) if p.suffix else p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    # Project is CPU-friendly; use CUDA if available.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_test_transform() -> transforms.Compose:
    # Match standard CIFAR-10 eval preprocessing.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])


def get_test_loader(cfg: dict) -> DataLoader:
    ds_cfg = cfg.get("dataset", {})
    root = ds_cfg.get("data_dir", "./data")
    num_workers = int(ds_cfg.get("num_workers", 2))
    batch_size = int(cfg.get("train", {}).get("batch_size", 128))

    test_set = datasets.CIFAR10(
        root=root, train=False, download=True, transform=default_test_transform()
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def build_model_from_cfg(cfg: dict) -> torch.nn.Module:
    """
    Prefer using your project model builder (model.py) if present.
    Fallback to torchvision ResNet-18 with 10 classes.
    """
    try:
        from model import build_model  # type: ignore
        model = build_model(num_classes=10)
    except Exception:
        from torchvision.models import resnet18
        model = resnet18(weights=None, num_classes=10)
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, device: torch.device) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")
    state = torch.load(ckpt_path, map_location=device)
    # Allow strict=False to tolerate key mismatches between training/eval builders.
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")


# ------------------------- plotting helpers ------------------------- #
def save_confusion_matrix(cm: np.ndarray, out_path: str | Path) -> None:
    out_path = ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticklabels(CIFAR10_CLASSES)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix → {out_path}")


def save_sample_grids(images: torch.Tensor,
                      labels: np.ndarray,
                      preds: np.ndarray,
                      out_dir: str | Path,
                      max_each: int = 32) -> None:
    out_dir = ensure_dir(out_dir)
    correct_idx = np.where(preds == labels)[0][:max_each]
    wrong_idx = np.where(preds != labels)[0][:max_each]

    def to_grid(idx: np.ndarray) -> torch.Tensor:
        if idx.size == 0:
            return None
        # de-normalize for visualization
        imgs = images[idx].clone().cpu()
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        imgs = imgs * std + mean
        grid = tvu.make_grid(imgs, nrow=8, padding=4)
        return grid

    cg = to_grid(correct_idx)
    wg = to_grid(wrong_idx)

    if cg is not None:
        tvu.save_image(cg, Path(out_dir) / "correct_grid.png")
    if wg is not None:
        tvu.save_image(wg, Path(out_dir) / "wrong_grid.png")
    print(f"Saved samples → {out_dir}")


# ------------------------- evaluation ------------------------- #
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, torch.Tensor]:
    model.eval().to(device)

    all_preds: List[int] = []
    all_labels: List[int] = []
    stash_imgs: List[torch.Tensor] = []

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(targets.numpy())

        # Keep a small stash for sample grids
        stash_imgs.append(imgs.cpu())

    preds_np = np.concatenate(all_preds, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    imgs_tensor = torch.cat(stash_imgs, dim=0)

    acc = (preds_np == labels_np).mean().item()
    cm = confusion_matrix(labels_np, preds_np, labels=list(range(10)))

    return float(acc), preds_np, labels_np, imgs_tensor, cm


def per_class_precision(cm: np.ndarray) -> List[float]:
    # precision = TP / (TP + FP) for each class
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    denom = tp + fp
    denom[denom == 0.0] = 1.0
    return (tp / denom).tolist()


# ------------------------- main ------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on CIFAR-10")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None, help="Override checkpoint path")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(args.seed))
    device = get_device()

    # Build model and load weights
    model = build_model_from_cfg(cfg)
    ckpt_path = args.ckpt or cfg.get("checkpoint", {}).get("best_path", "results/best.pt")
    load_checkpoint(model, ckpt_path, device=device)

    # Data
    loader = get_test_loader(cfg)

    # Eval
    acc, preds_np, labels_np, imgs_tensor, cm = evaluate(model, loader, device)
    precisions = per_class_precision(cm)

    # Reporting (cast to plain floats so f-strings with :.3f never fail)
    print("\nPer-class precision:")
    for i, name in enumerate(CIFAR10_CLASSES):
        p = float(precisions[i])
        print(f"{name:>10}: {p:.3f}")

    print(f"\nOverall accuracy: {float(acc):.3f}")

    worst_idx = int(np.argmin(np.array(precisions)))
    worst_prec = float(precisions[worst_idx])
    print(f"Worst class: {CIFAR10_CLASSES[worst_idx]} (precision={worst_prec:.3f})")

    # Outputs
    out_cfg = cfg.get("output", {})
    cm_path = out_cfg.get("cm", "results/confusion_matrix.png")
    samples_dir = out_cfg.get("samples_dir", "results/samples")

    save_confusion_matrix(cm, cm_path)
    save_sample_grids(imgs_tensor, labels_np, preds_np, samples_dir)


if __name__ == "__main__":
    main()
