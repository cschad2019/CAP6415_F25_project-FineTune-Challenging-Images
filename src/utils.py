"""
utils.py — helpers for seeding, plotting, paths, and metrics I/O.
"""
import os, json, random
import numpy as np
import torch
import matplotlib.pyplot as plt

def seed_everything(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    d = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def plot_curves(history: dict, out_path: str) -> None:
    ensure_dir(out_path)
    plt.figure()
    if "train_acc" in history: plt.plot(history["train_acc"], label="train_acc")
    if "val_acc"   in history: plt.plot(history["val_acc"],   label="val_acc")
    if "train_loss" in history: plt.plot(history["train_loss"], label="train_loss")
    if "val_loss"   in history: plt.plot(history["val_loss"],   label="val_loss")
    plt.xlabel("epoch"); plt.title("Accuracy/Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

def save_image_grid(tensors, titles, out_path: str, ncols: int = 8) -> None:
    ensure_dir(out_path)
    total = len(tensors); ncols = max(1, ncols)
    nrows = (total + ncols - 1) // ncols
    plt.figure(figsize=(ncols * 1.5, nrows * 1.5))
    for i, (img_t, title) in enumerate(zip(tensors, titles)):
        plt.subplot(nrows, ncols, i + 1)
        np_img = img_t.permute(1, 2, 0).cpu().numpy()
        np_img = np.clip(np_img, 0, 1)
        plt.imshow(np_img); plt.title(title, fontsize=8); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def save_json(obj: dict, out_path: str) -> None:
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
