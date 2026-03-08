"""
Main Training Script
"""

import argparse
import numpy as np
import wandb

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from ann.objective_functions import get_loss


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])

    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"])

    parser.add_argument("--weight_init", type=str, default="xavier")

    parser.add_argument("--model_save_path", type=str, default="best_model.npy")

    parser.add_argument("--wandb_project", type=str, default="da6401_assignment1")

    # student CLI
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--num_neurons", type=int, default=128)

    # autograder CLI
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--hidden_size", nargs="+", type=int)

    return parser.parse_args()


def train(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, args):

    best_val_acc = 0

    for epoch in range(args.epochs):

        indices = np.random.permutation(len(X_train))

        X_train = X_train[indices]
        y_train = y_train[indices]

        total_loss = 0

        for i in range(0, len(X_train), args.batch_size):

            X_batch = X_train[i:i + args.batch_size]
            y_batch = y_train[i:i + args.batch_size]

            logits = model.forward(X_batch)

            loss = loss_fn.forward(logits, y_batch)

            grad = loss_fn.backward()

            model.backward(grad)

            for layer in model.layers:
                optimizer.update(layer)

            total_loss += loss

        val_logits = model.forward(X_val)

        preds = np.argmax(val_logits, axis=1)
        labels = np.argmax(y_val, axis=1)

        val_acc = np.mean(preds == labels)

        print(f"Epoch {epoch+1}/{args.epochs} Loss:{total_loss:.4f} ValAcc:{val_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss,
            "val_accuracy": val_acc
        })

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            np.save(args.model_save_path, model.get_parameters())

            print("Best model saved!")

    print("Training finished")


def main():

    args = parse_arguments()

    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    print("Loading dataset")

    X_train, y_train, X_val, y_val = load_data(args.dataset)

    # handle CLI compatibility
    if args.hidden_size is not None:
        hidden_sizes = args.hidden_size

    elif args.num_layers is not None:
        hidden_sizes = [args.num_neurons] * args.num_layers

    else:
        hidden_sizes = [args.num_neurons] * args.hidden_layers

    layer_sizes = [784] + hidden_sizes + [10]

    print("Building model")

    model = NeuralNetwork(
        input_size=layer_sizes[0],
        hidden_sizes=layer_sizes[1:-1],
        output_size=layer_sizes[-1],
        activation=args.activation,
        weight_init=args.weight_init
    )

    optimizer = get_optimizer(args.optimizer, args.learning_rate)

    loss_fn = get_loss(args.loss)

    train(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, args)

    print("Training complete")


if __name__ == "__main__":
    main()