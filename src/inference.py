"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


def parse_arguments():
    # args: Command-line arguments for inference configuration, including model and config paths.

    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--config_path", type=str, default="best_config.json")
    parser.add_argument("--model_path", type=str, default="best_model.npy")

    return parser.parse_args()


def load_model(args):

    print("Building network architecture...")

    model = NeuralNetwork(args)

    print("Loading saved weights...")

    weights = np.load(args.model_path, allow_pickle=True).item()

    model.set_weights(weights)

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

    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")

    return accuracy, precision, recall, f1


def main():

    args = parse_arguments()

    # load config
    with open(args.config_path) as f:
        config = json.load(f)

    # merge config into args
    for key, value in config.items():
        setattr(args, key, value)

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