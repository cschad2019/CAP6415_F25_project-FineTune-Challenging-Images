<!-- README.md -->

# Fine-tuning a Small Model for Challenging Test Images (CAP6415 F25)

**Student:** Cale Schad (z23503021)  
**Course:** CAP6415 — Fall 2025  
**Project:** Individual Project #8 — *Fine-tune a small model for challenging test images*  
**Repository:** https://github.com/cschad2019/CAP6415_F25_project-FineTune-Challenging-Images

---

## Abstract

We train a small CNN (< 50 layers; ResNet-18) on CIFAR-10 to obtain a baseline model, identify the lowest-precision (most challenging) class from the confusion matrix, and then fine-tune using class-focused sampling and mild augmentations. The goal is to systematically improve the worst-performing class without significantly harming overall accuracy or model size. We report before/after precision for the target class, overall accuracy, confusion matrices, and qualitative sample grids. The code is fully reproducible on CPU via `requirements.txt`, a fixed random seed, and a single entry-point script.

---

## Environment & Key Settings

- **Python:** 3.12 (adjust if needed)  
- **Device:** CPU-only (no GPU required)  
- **Dataset:** CIFAR-10 (auto-downloads to `./data`)  
- **Model:** ResNet-18 (torchvision), < 50 layers  
- **Seed:** 42 for all experiments  

**Key config knobs (see `configs/`):**

| Setting              | Value (example)     |
|----------------------|---------------------|
| Baseline epochs      | 12                  |
| Fine-tune epochs     | 6                   |
| `focus_class`        | `cat`               |
| Oversample factor    | 4–6× for focus class|
| Freeze backbone      | `true` (fine-tune head) |

You can adjust these parameters in the YAML configs to explore other classes or training schedules.

---

## Motivation

Small CNNs are attractive for deployment (faster inference, lower memory) but often fail badly on a few specific classes, even if the overall accuracy looks good. This project asks:

> Instead of just reporting a single accuracy number, can we **systematically fix the worst class** with a targeted fine-tune?

---

## Objectives

1. Train a compact model (< 50 layers) on CIFAR-10 and identify the **lowest-precision** class from the test confusion matrix.  
2. Design a **class-focused fine-tuning** recipe using oversampling and mild augmentations to emphasize that challenging class.  
3. Quantify how much we can **lift that class’s precision** without significantly degrading performance on the remaining classes.

---

## Dataset

- **CIFAR-10**  
  - 50,000 train images / 10,000 test images  
  - 32×32 color images  
  - 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)  
- The dataset is downloaded automatically by torchvision into `./data` on first run.

---

## Results (Baseline vs. Targeted Fine-tune)

**Headline results (example run on CPU):**

| Run        | Overall Acc | Worst Class (precision) |
|------------|-------------|-------------------------|
| Baseline   | 0.761       | `cat` (0.536)           |
| Fine-tuned | 0.841       | `bird` (0.571)          |

- **Targeted improvement (cat):** 0.536 → **0.804** (+0.268).  
- After improving `cat`, the error profile shifts and `bird` becomes the new weakest class (0.571). This is expected when additional capacity is focused on a prior failure mode.

**Artifacts (file names in `results/`):**

- **Baseline**
  - `results/best.pt`
  - `results/baseline_confusion_matrix.png`
  - `results/samples_baseline/correct_grid.png`
  - `results/samples_baseline/wrong_grid.png`

- **Fine-tune**
  - `results/best_finetune.pt`
  - `results/finetune_confusion_matrix.png`
  - `results/samples_finetune/correct_grid.png`
  - `results/samples_finetune/wrong_grid.png`

These plots and grids are used in the project report to visualize where the model improves and which confusions remain.

---

## Method Overview

1. **Baseline training**
   - Train ResNet-18 on CIFAR-10 using standard augmentations (random crop/flip).  
   - Track train/validation metrics and save the best checkpoint (`results/best.pt`).  
   - Evaluate on the test set and compute per-class precision and a confusion matrix.

2. **Identify the hardest class**
   - Parse the per-class precision scores from the baseline evaluation.  
   - The lowest-precision class (in this run, `cat`) is selected as `focus_class`.

3. **Targeted fine-tuning**
   - Warm-start from the baseline checkpoint.  
   - Freeze most of the backbone and fine-tune the head (or last layers) to avoid catastrophic changes.  
   - Use a data loader that **oversamples the focus class** and uses mild augmentations.  
   - Train for a small number of additional epochs with a slightly reduced learning rate.

4. **Re-evaluation**
   - Evaluate the fine-tuned model on the full test set.  
   - Compare overall accuracy, per-class precision, and confusion matrices to quantify the trade-offs.

---

## Quick Start (Short Version)

For full details, see **[HOW_TO_RUN.md](HOW_TO_RUN.md)**.  

1. **Clone & cd:**

```bash
git clone https://github.com/cschad2019/CAP6415_F25_project-FineTune-Challenging-Images
cd CAP6415_F25_project-FineTune-Challenging-Images
