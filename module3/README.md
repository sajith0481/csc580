# Linear Regression Using TensorFlow (Assignment Package)

## Files
- `linear_regression_tf1.py` — Main script that generates data, builds TF v1-style graph, trains, prints results, and saves plots.
- `run_and_package.py` — Optional helper that runs the script and bundles outputs into `submission_linear_regression.zip`.

## How to Run
```bash
python linear_regression_tf1.py
```
Requires: Python 3.8+, TensorFlow (2.x is fine), NumPy, Matplotlib.

If you're using TensorFlow 2.x, the script uses `tf.compat.v1` to match the assignment's placeholders/session style.

## What to Submit
- A Word document with your introduction.
- A zip containing your Python code and **screenshots of your plots** (`training_data.png`, `fitted_line.png`). You can either paste the PNGs into the Word doc or include them in the zip.

## Notes
- Seeds are fixed to make results reproducible: `numpy=101`, `tensorflow=101`.
- Hyperparameters: `learning_rate=0.01`, `training_epochs=1000`.
- Outputs are saved in the same folder.
