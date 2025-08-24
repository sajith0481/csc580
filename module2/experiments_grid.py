#!/usr/bin/env python3
"""
experiments_grid.py
Runs a small grid of MNIST MLP experiments by invoking mnist_mlp_tf1_compat.py with different settings.
Writes a CSV summary of results to experiments/results.csv

NOTE: Adjust the grids as desired. Training multiple configs may take time on CPU.
"""
import os
import csv
import json
import time
import itertools
import subprocess
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()
TRAIN_SCRIPT = THIS_DIR / "mnist_mlp_tf1_compat.py"
OUT_DIR = THIS_DIR / "experiments"
(OUT_DIR / "runs").mkdir(parents=True, exist_ok=True)

# Default grids (edit freely)
HIDDEN_NODES = [128, 512]
HIDDEN_NODES2 = [0, 256]   # include 0 for single-layer, >0 for second layer
LEARNING_RATES = [0.5, 0.1, 0.05, 0.01]
BATCH_SIZES = [64, 100, 256]
EPOCHS = 15  # reduce a bit for runtime; increase for better accuracy
SEED = 42

def run_one(hn, hn2, lr, bs):
    tag = f"hn{hn}_hn2{hn2}_lr{lr}_bs{bs}"
    save_dir = OUT_DIR / "runs" / tag
    cmd = [
        "python", str(TRAIN_SCRIPT),
        "--epochs", str(EPOCHS),
        "--batch_size", str(bs),
        "--learning_rate", str(lr),
        "--hidden_nodes", str(hn),
        "--hidden_nodes2", str(hn2),
        "--seed", str(SEED),
        "--save_dir", str(save_dir)
    ]
    print("Running:", " ".join(map(str, cmd)))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0

    metrics_path = save_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        return dict(
            tag=tag,
            hidden_nodes=hn,
            hidden_nodes2=hn2,
            learning_rate=lr,
            batch_size=bs,
            epochs=EPOCHS,
            seed=SEED,
            test_accuracy=m.get("final_test_accuracy"),
            misclassified_count=m.get("misclassified_count"),
            save_dir=str(save_dir),
            secs=round(elapsed, 2)
        )
    else:
        return dict(
            tag=tag,
            hidden_nodes=hn,
            hidden_nodes2=hn2,
            learning_rate=lr,
            batch_size=bs,
            epochs=EPOCHS,
            seed=SEED,
            test_accuracy=None,
            misclassified_count=None,
            save_dir=str(save_dir),
            secs=round(elapsed, 2)
        )

def main():
    results = []
    for hn, hn2, lr, bs in itertools.product(HIDDEN_NODES, HIDDEN_NODES2, LEARNING_RATES, BATCH_SIZES):
        try:
            res = run_one(hn, hn2, lr, bs)
            results.append(res)
        except subprocess.CalledProcessError as e:
            print("Run failed:", e)
            results.append(dict(
                tag=f"hn{hn}_hn2{hn2}_lr{lr}_bs{bs}",
                hidden_nodes=hn, hidden_nodes2=hn2, learning_rate=lr, batch_size=bs,
                epochs=EPOCHS, seed=SEED, test_accuracy="RUN_FAILED", misclassified_count="RUN_FAILED",
                save_dir="", secs=0
            ))

    # Write CSV
    out_csv = OUT_DIR / "results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\nWrote:", out_csv)

    # Print best accuracy
    numeric = [r for r in results if isinstance(r["test_accuracy"], (int, float))]
    if numeric:
        best = max(numeric, key=lambda r: r["test_accuracy"])
        print(f"Best accuracy: {best['test_accuracy']:.4f} with config: {best['tag']}")
    else:
        print("No successful runs to report best accuracy.")

if __name__ == "__main__":
    main()
