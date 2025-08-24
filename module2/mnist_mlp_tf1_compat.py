#!/usr/bin/env python3
"""
mnist_mlp_tf1_compat.py
Configurable MNIST MLP trainer using TensorFlow v1-style graph APIs (compatible with TF2 via tf.compat.v1).
Implements the assignment pipeline and prints/exports answers for:
- Test accuracy
- Misclassified images (saved to disk)
- Effects of hidden neurons, learning rates, batch sizes, and an optional second hidden layer.

Usage examples:
  python mnist_mlp_tf1_compat.py --epochs 20 --batch_size 100 --learning_rate 0.5 --hidden_nodes 512
  python mnist_mlp_tf1_compat.py --hidden_nodes 256 --hidden_nodes2 128 --learning_rate 0.1 --batch_size 64

Outputs (under --save_dir):
  - metrics.json : parameters and final test accuracy
  - misclassified_indices.json : list of misclassified test indices
  - misclassified_grid.png : grid image of first 25 misclassified samples
  - sample_misclassified_*.png : first few individual misclassified samples

Author: (Your Name)
"""
import os
import json
import argparse
import datetime
import numpy as np
import tensorflow as tf

# Ensure TF2 users can run v1 graph code
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_and_prepare_data():
    # Load MNIST
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Flatten & scale
    train_images = x_train.reshape(60000, 784).astype('float32') / 255.0
    test_images  = x_test.reshape(10000, 784).astype('float32') / 255.0
    # One-hot labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
    y_test_oh  = tf.keras.utils.to_categorical(y_test, 10)
    return (train_images, y_train_oh), (test_images, y_test_oh), (x_test, y_test)


def build_graph(hidden_nodes=512, hidden_nodes2=0, learning_rate=0.5):
    g = tf.Graph()
    with g.as_default():
        input_images = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name="input_images")
        target_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, 10], name="target_labels")

        # Layer 1
        input_weights = tf.Variable(tf.random.truncated_normal([784, hidden_nodes], stddev=0.1), name="input_weights")
        input_biases = tf.Variable(tf.zeros([hidden_nodes]), name="input_biases")
        input_layer = tf.matmul(input_images, input_weights)
        hidden_layer = tf.nn.relu(input_layer + input_biases, name="hidden_relu_1")

        last_activation = hidden_layer
        last_dim = hidden_nodes

        # Optional second hidden layer
        if hidden_nodes2 and hidden_nodes2 > 0:
            hidden2_weights = tf.Variable(tf.random.truncated_normal([last_dim, hidden_nodes2], stddev=0.1), name="hidden2_weights")
            hidden2_biases  = tf.Variable(tf.zeros([hidden_nodes2]), name="hidden2_biases")
            hidden_layer2   = tf.nn.relu(tf.matmul(last_activation, hidden2_weights) + hidden2_biases, name="hidden_relu_2")
            last_activation = hidden_layer2
            last_dim = hidden_nodes2

        # Output layer (logits)
        hidden_weights = tf.Variable(tf.random.truncated_normal([last_dim, 10], stddev=0.1), name="hidden_weights")
        hidden_biases  = tf.Variable(tf.zeros([10]), name="hidden_biases")
        digit_weights  = tf.matmul(last_activation, hidden_weights) + hidden_biases

        # Loss + optimizer
        # Use non-v2 alias for wider TF compatibility
        loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=digit_weights, labels=target_labels))
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(digit_weights, 1), tf.argmax(target_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Softmax probabilities for inference
        probs = tf.nn.softmax(digit_weights, name="probs")

        init = tf.compat.v1.global_variables_initializer()

    return g, dict(
        input_images=input_images,
        target_labels=target_labels,
        optimizer=optimizer,
        accuracy=accuracy,
        probs=probs,
        init=init
    )


def train_and_evaluate(params):
    # Unpack
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    hidden_nodes = params["hidden_nodes"]
    hidden_nodes2 = params["hidden_nodes2"]
    learning_rate = params["learning_rate"]
    seed = params["seed"]
    save_dir = params["save_dir"]

    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)

    (x_train, y_train), (x_test, y_test), (x_test_imgs28, y_test_raw) = load_and_prepare_data()

    g, tensors = build_graph(hidden_nodes=hidden_nodes, hidden_nodes2=hidden_nodes2, learning_rate=learning_rate)

    with tf.compat.v1.Session(graph=g) as sess:
        sess.run(tensors["init"])

        TRAIN_DATASIZE = x_train.shape[0]
        PERIOD = TRAIN_DATASIZE // batch_size

        for e in range(epochs):
            idxs = np.random.permutation(TRAIN_DATASIZE)
            X_random = x_train[idxs]
            Y_random = y_train[idxs]

            for i in range(PERIOD):
                batch_X = X_random[i * batch_size:(i + 1) * batch_size]
                batch_Y = Y_random[i * batch_size:(i + 1) * batch_size]
                sess.run(tensors["optimizer"], feed_dict={tensors["input_images"]: batch_X, tensors["target_labels"]: batch_Y})

            acc = sess.run(tensors["accuracy"], feed_dict={tensors["input_images"]: x_test, tensors["target_labels"]: y_test})
            print(f"Training epoch {e + 1}  |  Test Accuracy: {acc:.4f}")

        # Final metrics
        final_acc = sess.run(tensors["accuracy"], feed_dict={tensors["input_images"]: x_test, tensors["target_labels"]: y_test})
        print(f"\nFINAL TEST ACCURACY: {final_acc:.4f}")

        # Predictions to find misclassifications
        probs = sess.run(tensors["probs"], feed_dict={tensors["input_images"]: x_test})
        preds = np.argmax(probs, axis=1)
        truth = np.argmax(y_test, axis=1)
        mis_idx = np.where(preds != truth)[0].tolist()

    # Save metrics and misclassified indices
    metrics = dict(
        params=params,
        final_test_accuracy=float(final_acc),
        misclassified_count=len(mis_idx),
        timestamp=str(datetime.datetime.now())
    )
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(save_dir, "misclassified_indices.json"), "w") as f:
        json.dump(mis_idx, f, indent=2)

    # Save a grid of first 25 misclassified samples
    if len(mis_idx) > 0:
        n = min(25, len(mis_idx))
        fig, axes = plt.subplots(5, 5, figsize=(8, 8))
        for i in range(n):
            idx = mis_idx[i]
            ax = axes[i // 5, i % 5]
            ax.imshow(x_test_imgs28[idx], cmap="gray_r")
            ax.set_title(f"Idx {idx}\nTrue {truth[idx]} Pred {preds[idx]}", fontsize=8)
            ax.axis("off")
        for j in range(25 - n):
            axes[(n + j) // 5, (n + j) % 5].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "misclassified_grid.png"), dpi=150)
        plt.close(fig)

        # Save first 5 individual misclassified samples
        for i in range(min(5, len(mis_idx))):
            idx = mis_idx[i]
            plt.figure()
            plt.imshow(x_test_imgs28[idx], cmap="gray_r")
            plt.title(f"Misclassified sample idx={idx} true={truth[idx]}")
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, f"sample_misclassified_{i+1}_idx_{idx}.png"), dpi=150, bbox_inches="tight")
            plt.close()

    print(f"Saved outputs to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="MNIST MLP Trainer (TF v1-compat)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument("--hidden_nodes", type=int, default=512)
    parser.add_argument("--hidden_nodes2", type=int, default=0, help="Set >0 to add a second hidden layer of this size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None, help="Directory for outputs. If None, autogenerated.")
    args = parser.parse_args()

    # Auto-generate save dir
    if args.save_dir is None:
        tag = f"hn{args.hidden_nodes}_hn2{args.hidden_nodes2}_lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}_seed{args.seed}"
        args.save_dir = os.path.join("outputs", tag)
    os.makedirs(args.save_dir, exist_ok=True)

    params = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_nodes=args.hidden_nodes,
        hidden_nodes2=args.hidden_nodes2,
        seed=args.seed,
        save_dir=args.save_dir
    )
    train_and_evaluate(params)


if __name__ == "__main__":
    main()
