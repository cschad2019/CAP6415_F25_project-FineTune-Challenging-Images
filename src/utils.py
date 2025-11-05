"""
utils.py
────────
Utility helpers: seeding for reproducibility, path creation, and plotting.
No external state; pure functions only.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def seed_everything(seed: int = 42) -> None:
    """Set seeds and deterministic flags for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    """
    Ensure the directory that would contain `path` exists.
    If `path` is a directory, we still handle it safely.
    """
    d = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def plot_curves(history: dict, out_path: str) -> None:
    """
    Plot training/validation accuracy curves.
    history: {"train_acc": [...], "val_acc": [...]}
    """
    ensure_dir(out_path)
    plt.figure()
    plt.plot(history.get("train_acc", []), label="train_acc")
    plt.plot(history.get("val_acc", []),   label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_image_grid(tensors, titles, out_path: str, ncols: int = 8) -> None:
    """
    Save a simple grid of small images with titles under each image.
    `tensors`: list of CxHxW tensors in [0,1]-ish (already normalized or de-normalized).
    `titles`:  list of short strings (same length as tensors).
    """
    ensure_dir(out_path)
    total = len(tensors)
    ncols = max(1, ncols)
    nrows = (total + ncols - 1) // ncols

    plt.figure(figsize=(ncols * 1.5, nrows * 1.5))
    for i, (img_t, title) in enumerate(zip(tensors, titles)):
        plt.subplot(nrows, ncols, i + 1)
        # Convert CxHxW -> HxWxC
        np_img = img_t.permute(1, 2, 0).cpu().numpy()
        # Clip to [0,1] for display if needed
        np_img = np.clip(np_img, 0, 1)
        plt.imshow(np_img)
        plt.title(title, fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
