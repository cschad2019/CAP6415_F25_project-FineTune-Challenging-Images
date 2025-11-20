"""
CAP6415 F25 - Project #8
Training entry point for CIFAR-10 with a small model (ResNet-18).

Purpose:
    Runs either a neutral *baseline* train or a *targeted fine-tune* (when the
    config enables a focus class / oversampling / partial freezing). Tracks
    accuracy/loss per epoch and saves the best checkpoint by validation accuracy.

Key CLI args:
    --config (str): Path to a YAML config (e.g., configs/baseline.yaml or
        configs/finetune_target_class.yaml).
    --seed (int): Random seed for reproducibility (e.g., 42).

Important config keys (see YAML):
    dataset.data_dir (str): Root folder where CIFAR-10 will be cached/downloaded.
    train.batch_size (int), train.epochs (int), train.lr (float)
    train.num_workers (int), train.pin_memory (bool)
    train.freeze_backbone (bool): If True, fine-tune only the classifier head.
    target.focus_class (str|None): Class name to bias training toward.
    target.oversample (bool), target.oversample_factor (int): Class rebalancing.

Outputs (written to ./results):
    - Baseline best model:        results/best.pt
    - Fine-tuned best model:      results/best_finetune.pt
    - (Optional) training curves/plots if enabled in utils.

Example:
    python src/train.py --config configs/baseline.yaml --seed 42
    python src/train.py --config configs/finetune_target_class.yaml --seed 42
"""

import argparse, os, random, yaml
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from data import get_dataloaders
from model import build_model  # your resnet18(<50 layers)


# ---------------------------- utils ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    # Use deterministic algorithms for reproducibility (may reduce speed)
    torch.use_deterministic_algorithms(True)


class LabelSmoothingCE(nn.Module):
    def __init__(self, eps: float = 0.0):
        super().__init__()
        self.eps = eps
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, logits, target):
        if self.eps <= 0.0:
            return nn.functional.cross_entropy(logits, target)
        log_probs = self.log_softmax(logits)
        n = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.eps / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def mixup_data(x, y, alpha=0.0):
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b, float(lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, agg_loss = 0, 0, 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        agg_loss += crit(logits, y).item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return correct / max(1, total), agg_loss / max(1, total)


def plot_curves(history, out_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    xs = list(range(1, len(history["train_acc"]) + 1))
    plt.plot(xs, history["train_acc"], label="train_acc")
    plt.plot(xs, history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    os.makedirs(Path(out_path).parent, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


class EMA:
    def __init__(self, model, decay=0.0):
        self.decay = float(decay)
        self.shadow = {}
        if self.decay > 0:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n] = p.detach().clone()

    def update(self, model):
        if self.decay <= 0:
            return
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def store(self, model):
        if self.decay <= 0:
            return {}
        backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        if self.decay <= 0:
            return
        for n, p in model.named_parameters():
            if p.requires_grad and n in backup:
                p.data.copy_(backup[n])


# ---------------------------- training ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", args.seed))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_loader, val_loader, _, class_names = get_dataloaders(cfg, eval_mode=False)

    # model
    model = build_model(num_classes=len(class_names))
    model.to(device)

    # warm start if provided
    init_ckpt = cfg.get("checkpoint", {}).get("init")
    if init_ckpt and Path(init_ckpt).exists():
        print(f"[info] warm-start from {init_ckpt}")
        state = torch.load(init_ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    # loss + opt + sched
    ls = float(cfg["train"].get("label_smoothing", 0.0))
    criterion = LabelSmoothingCE(eps=ls)

    name = cfg["optim"].get("name", "adamw").lower()
    lr = float(cfg["optim"].get("lr", 7e-4))
    wd = float(cfg["optim"].get("weight_decay", 5e-4))
    if name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(cfg["train"]["epochs"])
    warmup_epochs = int(cfg["optim"].get("warmup_epochs", 0))
    sched_name = cfg["optim"].get("scheduler", "cosine")

    if sched_name == "cosine":
        main_sched = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        if warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[warmup_epochs])
        else:
            scheduler = main_sched
    else:
        scheduler = None

    # options
    mixup_alpha = float(cfg["train"].get("mixup_alpha", 0.0))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
    ema_decay = float(cfg["train"].get("ema_decay", 0.0))
    ema = EMA(model, decay=ema_decay)

    best_path = cfg["checkpoint"]["best_path"]
    os.makedirs(Path(best_path).parent, exist_ok=True)

    history = {"train_acc": [], "val_acc": []}
    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            xm, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)

            logits = model(xm)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam) if mixup_alpha > 0 else criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            if mixup_alpha > 0:
                correct += (lam * (pred == y_a).sum().item() + (1 - lam) * (pred == y_b).sum().item())
            else:
                correct += (pred == y).sum().item()
            total += x.size(0)

            pbar.set_postfix(loss=f"{running_loss/max(1,total):.4f}")

        train_acc = correct / max(1, total)

        # validate (use EMA weights if enabled)
        backup = ema.store(model) if ema_decay > 0 else {}
        val_acc, _ = evaluate(model, val_loader, device)
        state_to_save = None
        if val_acc > best_val:
            best_val = val_acc
            state_to_save = {k: v.clone() for k, v in model.state_dict().items()}
        if ema_decay > 0:
            ema.restore(model, backup)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  | "
              f"train_loss={running_loss/max(1,total):.4f}")

        if state_to_save is not None:
            torch.save(state_to_save, best_path)
            print(f"Best val acc: {best_val:.4f} | saved -> {best_path}")

    # save curves
    plot_curves(history, cfg["output"]["curves"])


if __name__ == "__main__":
    main()
