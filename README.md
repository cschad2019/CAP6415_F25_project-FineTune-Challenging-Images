# Fine-tuning a Small Model for Challenging Test Images (CAP6415 F25)
**Student:** Cale Schad (z23503021)  
**Course:** CAP6415 — Fall 2025  
**Project:** Individual Project #8 — *Fine-tune a small model for challenging test images*  
**Repository:** https://github.com/cschad2019/CAP6415_F25_project-FineTune-Challenging-Images

## Abstract
We train a small CNN (< 50 layers; ResNet-18) on CIFAR-10 to obtain a baseline, identify the lowest-precision (most challenging) class from the confusion matrix, and fine-tune the model using class-focused sampling and mild augmentations. We report before/after precision for the target class, overall accuracy, curves, and qualitative samples. Code is fully reproducible via `requirements.txt` (or `env.yml`) and deterministic seeds.

---

## Results (baseline vs targeted fine-tune)

| Run        | Val Acc | Worst Class | Precision (worst) |
|------------|---------|-------------|-------------------|
| Baseline   | **0.6381** | **deer**    | **0.541**         |
| Fine-tuned | _fill_  | deer        | _fill_            |

See `results/*curves.png`, `results/*confusion_matrix.png`, and `results/samples_*`.

---

## About / How to Run

This project trains a small CNN on **CIFAR-10** to get a **baseline**, then runs a **targeted fine-tune** focused on the hardest class (by precision). Everything runs on **CPU**; artifacts are saved under `results/`.

### Environment (Windows / PowerShell)
```powershell
# From the repo root
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# If PowerShell blocks activation (policy):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

Run the baseline (train → eval)

# First eval will download CIFAR-10 to .\data\
python src\eval.py  --config configs\baseline.yaml --seed 42
python src\train.py --config configs\baseline.yaml --seed 42
python src\eval.py  --config configs\baseline.yaml --seed 42
 
 What to look for (terminal):

Per-class precision table and a line like:
Worst class: <name> (precision= …)

Example from my baseline: Val Acc ≈ 0.6381, worst class = deer, precision 0.541.

Where to see results (files):

results/curves.png — training/validation curves

results/confusion_matrix.png — confusion matrix

results/best.pt — baseline checkpoint

results/samples/correct_grid.png & results/samples/wrong_grid.png — qualitative samples

Run the fine-tune (after setting the worst class in configs/finetune_target_class.yaml)
python src\train.py --config configs\finetune_target_class.yaml --seed 42
python src\eval.py  --config configs\finetune_target_class.yaml --seed 42

What to look for (terminal):

Target class precision (e.g., deer) should improve vs baseline.

Overall accuracy should be similar or slightly better.

Where to see fine-tune results (files):

results/finetune_curves.png

results/finetune_confusion_matrix.png

results/best_finetune.pt

results/samples_finetune/*

Notes / Troubleshooting

If python opens the Microsoft Store, disable App execution aliases for python.exe / python3.exe in Windows Settings.

CPU-only Torch (no CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Quieter runs on CPU (optional):
$env:OMP_NUM_THREADS="2"; $env:MKL_NUM_THREADS="2"; python src\train.py --config configs\baseline.yaml --seed 42

Attribution

Built with PyTorch and torchvision; uses CIFAR-10 for research/education.
