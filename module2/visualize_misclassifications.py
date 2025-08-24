#!/usr/bin/env python3
"""
visualize_misclassifications.py
Quick viewer for a saved run's misclassified indices and images.
"""
import json, argparse, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Directory containing metrics.json and misclassified images")
    args = ap.parse_args()

    metrics_path = os.path.join(args.run_dir, "metrics.json")
    idx_path = os.path.join(args.run_dir, "misclassified_indices.json")
    grid_path = os.path.join(args.run_dir, "misclassified_grid.png")

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        print("Run params:", m["params"])
        print("Final test accuracy:", m["final_test_accuracy"])
        print("Misclassified count:", m["misclassified_count"])

    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idxs = json.load(f)
        print("First 20 misclassified indices:", idxs[:20])

    if os.path.exists(grid_path):
        img = mpimg.imread(grid_path)
        plt.figure(figsize=(8,8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Misclassified Images Grid")
        # Save instead of show for terminal environment
        output_path = os.path.join(args.run_dir, "misclassified_visualization.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Misclassified images grid saved to: {output_path}")
    else:
        print("No misclassified_grid.png found.")

if __name__ == "__main__":
    main()
