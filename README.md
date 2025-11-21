# CAP6415 F25 — Fine-tuning a Small Model for Challenging Test Images

**Student:** Cale Schad (z23503021)

## Abstract
Train a compact ResNet-18 on CIFAR-10 to get a baseline, find the weakest class via per-class precision/confusion matrix, then fine-tune with class-focused oversampling/augmentations to improve that class without hurting overall accuracy. Runs on CPU; configs live in `configs/`.

## Environment
- Python 3.12 (3.11+ should work)
- CPU or CUDA (CPU tested)
- Data: CIFAR-10 auto-downloads to `./data`
- Dependencies: `pip install -r requirements.txt`

## Quickstart
```bash
# setup
python -m venv .venv && .\.venv\Scripts\activate   # on Windows PowerShell
pip install -r requirements.txt

# baseline train
python src\train.py --config configs\baseline.yaml --seed 42
# checkpoint: results/best.pt

# targeted fine-tune (focus class in config)
python src\train.py --config configs\finetune_target_class.yaml --seed 42
# checkpoint: results/best_finetune.pt

# evaluate a run (uses test set)
python src\eval.py --config configs\baseline.yaml
python src\eval.py --config configs\finetune_target_class.yaml
# outputs: confusion matrices + sample grids in results/
```

## Reproducibility
- Seeds set in code (`seed` in config/CLI) and `torch.use_deterministic_algorithms(True)` enabled; expect deterministic kernels where supported. Flip to `False` if your hardware complains.
- No external data beyond CIFAR-10; paths use repo-relative defaults.
- Checkpoints are saved from the evaluated weights (EMA-aware) so reported best metrics match the saved model.

## Files
- `src/train.py` — training loop with mixup, EMA, checkpointing.
- `src/eval.py` — evaluation, per-class precision, confusion matrix, sample grids.
- `src/data.py` — CIFAR-10 loaders, optional oversampling of focus class.
- `configs/*.yaml` — baseline and fine-tune configs.
- `results/` — checkpoints and figures (created on run).
- `How to run.md` — step-by-step setup and usage.
- `week[1-5]log.txt` — weekly logs per project requirement.

## Results (current)
- **Baseline (3 epochs, CPU):** overall accuracy 0.761; worst class `cat` precision 0.536. Per-class: airplane 0.800, automobile 0.928, bird 0.614, cat 0.536, deer 0.797, dog 0.669, frog 0.722, horse 0.890, ship 0.876, truck 0.904.
- **Fine-tune (8 epochs, CPU):** overall accuracy 0.839; worst class `bird` precision 0.578. Per-class: airplane 0.904, automobile 0.931, bird 0.578, cat 0.838, deer 0.883, dog 0.790, frog 0.920, horse 0.876, ship 0.933, truck 0.927.
- Targeted lift: cat precision improved from 0.536 -> 0.838; new weakest class shifts to bird.
- Artifacts: baseline -> `results/baseline_confusion_matrix.png`, `results/samples_baseline/`; fine-tune -> `results/finetune_confusion_matrix.png`, `results/samples_finetune/`; checkpoints `results/best.pt` and `results/best_finetune.pt`.

## Runtime (CPU)
- Baseline (3 epochs, batch 128): ~30 minutes.
- Fine-tune (8 epochs, batch 128): ~80–90 minutes.

## Attribution
- Model backbone: torchvision ResNet-18.
- Dataset: CIFAR-10 (MIT License via torchvision).
- Techniques: mixup, EMA, cosine LR, oversampling.
