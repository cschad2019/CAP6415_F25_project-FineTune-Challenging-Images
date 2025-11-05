"""
train.py
────────
Train a small CNN on CIFAR-10 using a YAML config.
Saves:
- Best checkpoint → `checkpoint.best_path`
- Accuracy curves → `output.curves`
"""

import argparse
import yaml
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from utils import seed_everything, plot_curves, ensure_dir
from data import get_dataloaders
from model import build_model

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute top-1 accuracy for a batch."""
    preds = logits.argmax(1)
    return (preds == y).float().mean().item()

def main():
    # ─── CLI ───────────────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = ap.parse_args()

    # ─── Load Config & Seed ────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = args.seed
    seed_everything(args.seed)

    # ─── Data ──────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _, class_names = get_dataloaders(cfg, eval_mode=False)

    # ─── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=cfg["train"].get("freeze_backbone", False)
    ).to(device)

    # Optional warm start
    init_ckpt = cfg["checkpoint"].get("init")
    if init_ckpt and os.path.exists(init_ckpt):
        model.load_state_dict(torch.load(init_ckpt, map_location=device), strict=False)

    # Loss / Optimizer
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = Adam(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

    # ─── Train Loop ────────────────────────────────────────────────────────────
    history = {"train_acc": [], "val_acc": []}
    best_val = 0.0
    epochs = int(cfg["train"]["epochs"])

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running_train_acc = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running_train_acc += accuracy(logits, y)
        train_acc = running_train_acc / len(train_loader)

        # Validate
        model.eval()
        running_val_acc = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                running_val_acc += accuracy(logits, y)
        val_acc = running_val_acc / len(val_loader)

        # Log & Save
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_path = cfg["checkpoint"]["best_path"]
            ensure_dir(best_path)
            torch.save(model.state_dict(), best_path)

    # Curves
    plot_curves(history, cfg["output"]["curves"])
    print(f"Best val acc: {best_val:.4f} | saved → {cfg['checkpoint']['best_path']}")

if __name__ == "__main__":
    main()
