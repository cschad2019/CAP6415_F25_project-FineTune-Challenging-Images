# scripts/run_pipeline.py
import argparse, subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]   # repo root
PY = sys.executable                                  # current VS Code interpreter (.venv)

def run(cmd):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    r.check_returncode()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "finetune"], default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "baseline":
        cfg = "configs/baseline.yaml"
    else:
        cfg = "configs/finetune_target_class.yaml"

    # Train → Eval
    run([PY, "src/train.py", "--config", cfg, "--seed", str(args.seed)])
    run([PY, "src/eval.py",  "--config", cfg, "--seed", str(args.seed)])

if __name__ == "__main__":
    main()
