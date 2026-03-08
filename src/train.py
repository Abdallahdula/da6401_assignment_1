"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb
import json

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


def parse_arguments():
    # args: Command-line arguments for training configuration, including dataset, epochs, batch size, etc.

    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=0.0001)

    parser.add_argument("--optimizer", type=str, default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])

    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--hidden_size", type=int, nargs="+", default=[128, 128, 128])

    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"])

    parser.add_argument("--weight_init", type=str, default="xavier")

    parser.add_argument("--model_save_path", type=str,
                        default="best_model.npy")

    return parser.parse_args()


def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]


def train(model, X_train, y_train, X_val, y_val, args):

    best_val_acc = 0
    n = X_train.shape[0]

    # Initialize W&B run
    wandb.init()

    # Log hyperparameters
    config = wandb.config
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.optimizer = args.optimizer
    config.num_layers = args.num_layers
    config.hidden_size = args.hidden_size

    for epoch in range(args.epochs):

        indices = np.random.permutation(n)

        X_train = X_train[indices]
        y_train = y_train[indices]

        total_loss = 0

        for i in range(0, n, args.batch_size):

            X_batch = X_train[i:i + args.batch_size]
            y_batch = y_train[i:i + args.batch_size]

            if len(y_batch.shape) == 1:
                y_batch = one_hot(y_batch)

            logits = model.forward(X_batch)

            loss = model.loss_fn.forward(logits, y_batch)

            model.backward(y_batch, logits)

            model.update_weights()

            total_loss += loss

        # validation preparation
        if len(y_val.shape) == 1:
            y_val_eval = one_hot(y_val)
        else:
            y_val_eval = y_val

        val_acc = model.evaluate(X_val, y_val_eval)

        print(f"Epoch {epoch+1}/{args.epochs}  Loss: {total_loss:.4f}  Val Acc: {val_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss,
            "val_accuracy": val_acc
        })

        # Log validation accuracy
        wandb.log({"val_accuracy": val_acc})

        # save best model
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            weights = model.get_weights()

            np.save(args.model_save_path, weights, allow_pickle=True)

            wandb.save(args.model_save_path)

            print("Best model saved!")

    print("Training finished!")


def save_best_config(args):

    config_path = "best_config.json"

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"Best config saved to {config_path}")


def main():

    args = parse_arguments()

    wandb.init(
        project="da6401-assignment1",
        config=vars(args)
    )

    print("Configuration:")
    print(args)

    print("Loading dataset...")

    X_train, y_train, X_val, y_val = load_data(args.dataset)

    print("Building neural network...")

    model = NeuralNetwork(args)

    print("Starting training...")

    train(model, X_train, y_train, X_val, y_val, args)

    # Save best config for submission
    save_best_config(args)

    wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()