# How to Run

Step-by-step guide to reproduce the baseline and targeted fine-tune runs on CIFAR-10.

## 1) Prereqs
- Python 3.11+ (tested 3.12)
- Git
- ~200MB disk for CIFAR-10 download
- Windows PowerShell commands shown; adapt paths for macOS/Linux.

## 2) Clone and enter repo
```bash
git clone https://github.com/cschad2019/CAP6415_F25_project-FineTune-Challenging-Images
cd CAP6415_F25_project-FineTune-Challenging-Images
```

## 3) Create venv and install deps
```bash
python -m venv .venv
.\.venv\Scripts\activate   # on PowerShell; use source .venv/bin/activate for bash
pip install -r requirements.txt
```

## 4) Run baseline training
```bash
python src\train.py --config configs\baseline.yaml --seed 42
```
Outputs:
- Checkpoint: `results/best.pt`
- Curve plot: `results/baseline_curves.png`

## 5) Run targeted fine-tune (focus class in config)
```bash
python src\train.py --config configs\finetune_target_class.yaml --seed 42
```
Outputs:
- Checkpoint: `results/best_finetune.pt`
- Curve plot: `results/finetune_curves.png`
- Final fine-tune metrics (CPU run): overall acc 0.839; worst class `bird` precision 0.578; cat precision 0.838.

## 6) Evaluate
```bash
python src\eval.py --config configs\baseline.yaml
python src\eval.py --config configs\finetune_target_class.yaml
```
Outputs (into `results/`): confusion matrices and sample grids (`samples_*`). Use `--ckpt` to override checkpoint path if needed.

## Current metrics (CPU runs)
- Baseline: overall acc 0.761; worst class `cat` precision 0.536.
- Fine-tune: overall acc 0.839; worst class `bird` precision 0.578; cat precision 0.838.

## Reproducibility notes
- Seeds come from config/CLI; `torch.use_deterministic_algorithms(True)` is on. If a deterministic kernel is unavailable, PyTorch will raise; you can flip it to False for speed/compatibility but runs may vary slightly.
- Data auto-downloads to `./data`; no extra assets needed.
- Dataloaders avoid training-time oversampling/augmentations when `eval_mode=True`.

## Troubleshooting
- If you hit CUDA/cuDNN determinism errors, set `torch.use_deterministic_algorithms(False)` in `src/train.py` or run on CPU.
- Delete `./data/cifar-10-python.tar.gz` and rerun if download corrupts.
