#!/usr/bin/env python3
"""
tox21_tf1.py
Neural network for Tox21 (binary classification, first task) using TensorFlow v1-style APIs.
Implements:
- Data loading via DeepChem (MolNet Tox21)
- Placeholders, a hidden layer with ReLU + Dropout
- Sigmoid output, cross-entropy loss
- Adam optimizer
- Mini-batch training with TensorBoard summaries
- Accuracy metric on validation and test sets
- Saves metrics to metrics.txt

Run:
  python tox21_tf1.py --logdir ./runs/fcnet-tox21 --epochs 10 --batch_size 100 --learning_rate 0.001 --n_hidden 50 --dropout 0.5

Note:
  Requires: tensorflow (2.x OK via tf.compat.v1), deepchem, numpy, scikit-learn, matplotlib (for optional plots)
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Disable TF2 eager; use v1 graph API to mirror assignment
tf1 = tf.compat.v1
tf1.disable_eager_execution()

# Fixed seeds
np.random.seed(456)
try:
    tf.set_random_seed(456)  # TF1
except Exception:
    tf1.set_random_seed(456)  # TF2 compat

# Optional matplotlib for local plotting of loss curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_data():
    import deepchem as dc
    _, (train, valid, test), _ = dc.molnet.load_tox21()
    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X,  test_y,  test_w  = test.X,  test.y,  test.w

    # Keep only the first task
    train_y = train_y[:, 0]
    valid_y = valid_y[:, 0]
    test_y  = test_y[:, 0]
    train_w = train_w[:, 0]
    valid_w = valid_w[:, 0]
    test_w  = test_w[:, 0]
    return (train_X, train_y, train_w), (valid_X, valid_y, valid_w), (test_X, test_y, test_w)

def build_graph(d, n_hidden, learning_rate):
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope("placeholders"):
            x = tf1.placeholder(tf.float32, (None, d), name="x")
            y = tf1.placeholder(tf.float32, (None,), name="y")
            keep_prob = tf1.placeholder_with_default(1.0, shape=(), name="keep_prob")

        with tf.name_scope("hidden-layer"):
            W1 = tf1.Variable(tf.random.normal((d, n_hidden)), name="W1")
            b1 = tf1.Variable(tf.random.normal((n_hidden,)), name="b1")
            x_hidden = tf.nn.relu(tf.matmul(x, W1) + b1, name="relu1")
            # Dropout
            x_hidden_drop = tf.nn.dropout(x_hidden, rate=1.0-keep_prob, name="dropout1")

        with tf.name_scope("output"):
            W2 = tf1.Variable(tf.random.normal((n_hidden, 1)), name="W2")
            b2 = tf1.Variable(tf.random.normal((1,)), name="b2")
            y_logit = tf.matmul(x_hidden_drop, W2) + b2
            y_one_prob = tf.sigmoid(y_logit, name="y_one_prob")
            y_pred = tf.round(y_one_prob, name="y_pred")

        with tf.name_scope("loss"):
            y_expand = tf.expand_dims(y, 1)
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
            l = tf.reduce_sum(entropy, name="loss_sum")

        with tf.name_scope("optim"):
            train_op = tf1.train.AdamOptimizer(learning_rate).minimize(l)

        with tf.name_scope("summaries"):
            tf1.summary.scalar("loss", l)
            merged = tf1.summary.merge_all()

        init = tf1.global_variables_initializer()

    tensors = dict(x=x, y=y, keep_prob=keep_prob, y_pred=y_pred, y_one_prob=y_one_prob,
                   loss=l, train_op=train_op, merged=merged, init=init)
    return g, tensors

def iterate_minibatches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        end = start + batch_size
        excerpt = indices[start:end]
        yield X[excerpt], y[excerpt]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="./runs/fcnet-tox21")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=0.001)
    ap.add_argument("--n_hidden", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.5, help="keep_prob during training = 1 - dropout_rate")
    args = ap.parse_args()

    (train_X, train_y, _), (valid_X, valid_y, _), (test_X, test_y, _) = load_data()
    d = train_X.shape[1]

    g, t = build_graph(d=d, n_hidden=args.n_hidden, learning_rate=args.learning_rate)

    run_dir = args.logdir
    os.makedirs(run_dir, exist_ok=True)
    writer = tf1.summary.FileWriter(run_dir)

    losses = []
    step = 0
    with tf1.Session(graph=g) as sess:
        sess.run(t["init"])
        writer.add_graph(sess.graph)

        for epoch in range(args.epochs):
            for batch_X, batch_y in iterate_minibatches(train_X, train_y, args.batch_size, shuffle=True):
                feed = {t["x"]: batch_X, t["y"]: batch_y, t["keep_prob"]: 1.0 - args.dropout}
                _, summary, loss_val = sess.run([t["train_op"], t["merged"], t["loss"]], feed_dict=feed)
                writer.add_summary(summary, step)
                step += 1
                losses.append(loss_val)

            # Validation progress per epoch
            v_pred = sess.run(t["y_pred"], feed_dict={t["x"]: valid_X})
            v_acc = accuracy_score(valid_y, v_pred)
            print(f"Epoch {epoch+1}/{args.epochs}  |  Valid Accuracy: {v_acc:.4f}  |  Last batch loss: {loss_val:.2f}")

        # Final metrics
        v_pred = sess.run(t["y_pred"], feed_dict={t["x"]: valid_X})
        t_pred = sess.run(t["y_pred"], feed_dict={t["x"]: test_X})
        v_acc = accuracy_score(valid_y, v_pred)
        t_acc = accuracy_score(test_y, t_pred)

    # Save metrics and a loss curve image
    with open(os.path.join(os.getcwd(), "metrics.txt"), "w") as f:
        f.write("Tox21 NN (first task)\n")
        f.write(f"Validation accuracy: {v_acc:.4f}\n")
        f.write(f"Test accuracy: {t_acc:.4f}\n")

    # Save loss curve
    if losses:
        plt.figure()
        plt.plot(losses)
        plt.title("Training Loss (sum of cross-entropy per batch)")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "loss_curve.png"), dpi=150)
        plt.close()

    print("Saved metrics to metrics.txt and loss curve to loss_curve.png")
    print(f"TensorBoard logdir: {run_dir}\nLaunch with: tensorboard --logdir {run_dir}")

if __name__ == "__main__":
    main()
