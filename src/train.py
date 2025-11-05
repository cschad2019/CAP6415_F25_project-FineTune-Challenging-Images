"""
train.py — CIFAR-10 training with optional warm-start, early stopping, and loss/acc curves.
"""
import argparse, os, yaml
import torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utils import seed_everything, plot_curves, ensure_dir, save_json
from data import get_dataloaders
from model import build_model

def batch_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = args.seed
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _, class_names = get_dataloaders(cfg, eval_mode=False)

    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=cfg["train"].get("freeze_backbone", False)
    ).to(device)

    # optional warm start
    init_ckpt = cfg["checkpoint"].get("init")
    if init_ckpt and os.path.exists(init_ckpt):
        try:
            model.load_state_dict(torch.load(init_ckpt, map_location=device), strict=False)
            print(f"[info] warm-start from {init_ckpt}")
        except Exception as e:
            print(f"[warn] failed to load init checkpoint ({e}); training from scratch.")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)))
    opt = Adam(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    scheduler = StepLR(opt, step_size=max(1, int(cfg["train"].get("lr_step", 0) or 0)), gamma=float(cfg["train"].get("lr_gamma", 0.1))) \
                if cfg["train"].get("lr_step") else None

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val, epochs = 0.0, int(cfg["train"]["epochs"])
    patience = cfg["train"].get("early_stopping_patience", None)
    bad_epochs = 0

    try:
        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            tr_acc = tr_loss = 0.0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward(); opt.step()
                tr_loss += loss.item()
                tr_acc  += batch_accuracy(logits, y)
            train_loss = tr_loss / len(train_loader)
            train_acc  = tr_acc  / len(train_loader)

            # Val
            model.eval()
            va_acc = va_loss = 0.0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    va_loss += loss.item()
                    va_acc  += batch_accuracy(logits, y)
            val_loss = va_loss / len(val_loader)
            val_acc  = va_acc  / len(val_loader)

            if scheduler: scheduler.step()

            history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)
            history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
            print(f"Epoch {epoch}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                  f"| train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            # Save best
            if val_acc > best_val:
                best_val = val_acc
                best_path = cfg["checkpoint"]["best_path"]
                ensure_dir(best_path); torch.save(model.state_dict(), best_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if patience is not None and bad_epochs >= int(patience):
                    print(f"[early-stop] no val improvement for {patience} epoch(s).")
                    break

        # Curves + train summary
        plot_curves(history, cfg["output"]["curves"])
        save_json(
            {"project": cfg.get("project_name","run"),
             "best_val_acc": best_val,
             "epochs_ran": len(history["val_acc"]),
             "seed": cfg["seed"]},
            os.path.join("results", f"{cfg.get('project_name','run')}_train_metrics.json")
        )
        print(f"Best val acc: {best_val:.4f} | saved → {cfg['checkpoint']['best_path']}")

    except KeyboardInterrupt:
        print("\n[info] interrupted; saving partial progress.")
        best_path = cfg["checkpoint"]["best_path"]
        ensure_dir(best_path); torch.save(model.state_dict(), best_path)
        plot_curves(history, cfg["output"]["curves"])

if __name__ == "__main__":
    main()
