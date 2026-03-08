"""
Inference Script
Evaluate trained models
"""

import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


def parse_arguments():

    parser = argparse.ArgumentParser(description="Run inference")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--model_path", type=str, default="best_model.npy")

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--hidden_size", type=int, nargs="+",
                        default=[128, 128])

    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("--weight_init", type=str, default="xavier")

    return parser.parse_args()


def load_model(args):

    print("Loading saved weights...")

    weights = np.load(args.model_path, allow_pickle=True).item()

    # Detect architecture from weights
    hidden_sizes = []
    i = 0

    while f"W{i}" in weights:
        hidden_sizes.append(weights[f"W{i}"].shape[1])
        i += 1

    # Last layer is output layer
    output_size = hidden_sizes[-1]
    hidden_sizes = hidden_sizes[:-1]

    print("Building network architecture...")

    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=args.activation,
        weight_init=args.weight_init
    )

    print("Assigning weights...")

    for i, layer in enumerate(model.layers):
        layer.W = weights[f"W{i}"]
        layer.b = weights[f"b{i}"]

    return model


def evaluate(model, X_test, y_test, batch_size):

    predictions = []
    labels = []

    n = X_test.shape[0]

    for i in range(0, n, batch_size):

        X_batch = X_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]

        logits = model.forward(X_batch)

        pred = np.argmax(logits, axis=1)

        if len(y_batch.shape) > 1:
            label = np.argmax(y_batch, axis=1)
        else:
            label = y_batch

        predictions.extend(pred)
        labels.extend(label)

    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracy = np.mean(predictions == labels)

    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

    return accuracy, precision, recall, f1


def main():

    args = parse_arguments()

    wandb.init(project="da6401-assignment1-inference")

    print("Loading dataset...")

    _, _, X_test, y_test = load_data(args.dataset)

    print("Loading trained model...")

    model = load_model(args)

    print("Running inference...")

    accuracy, precision, recall, f1 = evaluate(
        model,
        X_test,
        y_test,
        args.batch_size
    )

    print("\nEvaluation Results")
    print("------------------")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    wandb.log({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    wandb.finish()


if __name__ == "__main__":
    main()