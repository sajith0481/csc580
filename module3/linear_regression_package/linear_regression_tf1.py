#!/usr/bin/env python3
"""
linear_regression_tf1.py
TensorFlow (v1-style) linear regression on synthetic noisy line data.
- Plots training data
- Builds TF graph with placeholders X, Y; trainable W, b
- Uses MSE cost and SGD optimizer
- Trains for 1000 epochs
- Prints final cost, W, b
- Saves fitted line plot

Run:
  python linear_regression_tf1.py

Outputs (in the same folder):
  - training_data.png
  - fitted_line.png
  - results.txt
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

# Use TF v1 graph mode for placeholder/session semantics
tf1 = tf.compat.v1
tf1.disable_eager_execution()

# Set seeds
np.random.seed(101)
try:
    tf.set_random_seed(101)  # for TF 1.x
except Exception:
    tf1.set_random_seed(101) # for TF 2.x compat

# Generate synthetic linear data with noise
x = np.linspace(0, 50, 50)
y = 2 * x + 3 + np.random.uniform(-4, 4, 50)  # y = 2x + 3 + noise

# Normalize data to prevent numerical instability
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)
n = len(x)

# 1) Plot training data
plt.figure()
plt.scatter(x, y, s=30)
plt.title("Training Data (noisy linear)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("training_data.png", dpi=150)
plt.close()

# 2) Placeholders
X = tf1.placeholder(tf.float32, name="X")
Y = tf1.placeholder(tf.float32, name="Y")

# 3) Trainable variables
W = tf1.Variable(np.random.randn(), dtype=tf.float32, name="weight")
b = tf1.Variable(np.random.randn(), dtype=tf.float32, name="bias")

# 4) Hyperparameters
learning_rate = 0.0001  # Further reduced to prevent numerical instability
training_epochs = 1000

# 5) Hypothesis, cost, optimizer
y_pred = W * X + b                          # hypothesis
cost = tf1.reduce_mean(tf.square(Y - y_pred))  # MSE
optimizer = tf1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 6) Training
with tf1.Session() as sess:
    sess.run(tf1.global_variables_initializer())

    for epoch in range(training_epochs):
        for xi, yi in zip(x, y):
            sess.run(optimizer, feed_dict={X: xi, Y: yi})

        # Optional: print a few progress updates
        if (epoch+1) % 100 == 0:
            c = sess.run(cost, feed_dict={X: x, Y: y})
            w_val, b_val = sess.run([W, b])
            print(f"Epoch {epoch+1:4d}  cost={c:.4f}  W={w_val:.4f}  b={b_val:.4f}")

    # 7) Final results
    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    final_W, final_b = sess.run([W, b])

    print("\n=== Final Results ===")
    print(f"Training cost: {training_cost:.6f}")
    print(f"Weight (W): {final_W:.6f}")
    print(f"Bias   (b): {final_b:.6f}")

    # Save to text file
    with open("results.txt", "w") as f:
        f.write("Linear Regression Results (TensorFlow v1-style)\n")
        f.write(f"Training cost: {training_cost:.6f}\n")
        f.write(f"W: {final_W:.6f}\n")
        f.write(f"b: {final_b:.6f}\n")

    # 8) Plot fitted line
    y_hat = final_W * x + final_b
    plt.figure()
    plt.scatter(x, y, s=30, label="Data")
    plt.plot(x, y_hat, label="Fitted line")
    plt.title("Linear Regression Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fitted_line.png", dpi=150)
    plt.close()
