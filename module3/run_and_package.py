#!/usr/bin/env python3
"""
run_and_package.py
Helper to run the linear regression script and zip outputs for submission.

This requires TensorFlow installed. If TF isn't available, it will still create a zip
with code (without plots).

Outputs:
  - submission_linear_regression.zip
"""
import os, zipfile, subprocess, sys

root = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(root, "linear_regression_tf1.py")
zip_path = os.path.join(root, "submission_linear_regression.zip")

# Try to run the training script to generate plots
ran = False
try:
    print("Running linear_regression_tf1.py ...")
    subprocess.run([sys.executable, script], check=True, cwd=root)
    ran = True
except Exception as e:
    print("Could not run training (likely missing TensorFlow). Proceeding to zip code only.")
    print(e)

# Collect files for zip
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(script, arcname="linear_regression_tf1.py")
    # Include outputs if present
    for fname in ["training_data.png", "fitted_line.png", "results.txt"]:
        fpath = os.path.join(root, fname)
        if os.path.exists(fpath):
            z.write(fpath, arcname=fname)

print("Wrote:", zip_path)
