"""
eval.py
───────
Evaluate a trained (or randomly initialized) model on CIFAR-10 test set.
Outputs:
- Per-class precision (prints worst class for fine-tuning target)
- Confusion matrix image → `output.cm`
- Sample grids (correct / incorrect) → `output.samples_dir`
"""

import argparse
import yaml
import os
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score

from utils import seed_everything, ensure_dir, save_image_grid
from data import get_dataloaders
from model import build_model

def _plot_confusion_matrix(cm: np.ndarray, classes: List[str], out_path: str) -> None:
    """Save a simple confusion matrix heatmap (no seaborn)."""
    ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _denorm_for_viz(x: torch.Tensor) -> torch.Tensor:
    """
    Roughly de-normalize CIFAR-10 (for visualization only).
    Assumes Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)).
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(3,1,1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010], device=x.device).view(3,1,1)
    return x * std + mean

def main():
    # ─── CLI ───────────────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ─── Load Config & Seed ────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = args.seed
    seed_everything(args.seed)

    # ─── Data ──────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader, class_names = get_dataloaders(cfg, eval_mode=True)

    # ─── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    ckpt = cfg["checkpoint"].get("best_path")
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        print(f"Loaded checkpoint: {ckpt}")
    model.eval()

    # ─── Inference ─────────────────────────────────────────────────────────────
    preds, labels = [], []
    correct_imgs, correct_titles = [], []
    wrong_imgs, wrong_titles = [], []

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            y_hat = logits.argmax(1).cpu()

            preds.append(y_hat.numpy())
            labels.append(y.numpy())

            # Collect a few samples for visualization (first N per batch)
            for i in range(min(8, x.shape[0])):  # keep small per batch
                gt = class_names[y[i].item()]
                pr = class_names[y_hat[i].item()]
                img = _denorm_for_viz(x[i])
                if gt == pr and len(correct_imgs) < 32:
                    correct_imgs.append(img)
                    correct_titles.append(f"✓ {pr}")
                elif gt != pr and len(wrong_imgs) < 32:
                    wrong_imgs.append(img)
                    wrong_titles.append(f"x {pr} (gt:{gt})")

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)

    # ─── Metrics / CM ──────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    _plot_confusion_matrix(cm, class_names, cfg["output"]["cm"])

    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    worst_idx = int(np.argmin(per_prec))

    print("\nPer-class precision:")
    for i, p in enumerate(per_prec):
        print(f"{class_names[i]:>10s}: {p:.3f}")
    print(f"\nWorst class: {class_names[worst_idx]} (precision={per_prec[worst_idx]:.3f})")
    print(f"Saved confusion matrix → {cfg['output']['cm']}")

    # ─── Save sample grids ─────────────────────────────────────────────────────
    samples_dir = cfg["output"].get("samples_dir", "results/samples")
    ensure_dir(samples_dir)
    if correct_imgs:
        save_image_grid(correct_imgs, correct_titles, os.path.join(samples_dir, "correct_grid.png"))
    if wrong_imgs:
        save_image_grid(wrong_imgs, wrong_titles, os.path.join(samples_dir, "wrong_grid.png"))
    print(f"Saved samples → {samples_dir}")

if __name__ == "__main__":
    main()
