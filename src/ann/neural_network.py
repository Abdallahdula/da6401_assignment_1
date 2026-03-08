"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import Dense
from .activations import ReLU, Sigmoid, Tanh
from .objective_functions import MeanSquaredError, CrossEntropyLoss


class NeuralNetwork:

    def __init__(self, cli_args):
        # args: Command-line arguments containing hyperparameters such as learning rate, number of layers, activation function, etc.

        self.layers = []
        self.activations = []

        input_size = 784
        hidden_sizes = cli_args.hidden_size
        activation = cli_args.activation
        weight_init = cli_args.weight_init

        sizes = [input_size] + hidden_sizes + [10]

        for i in range(len(sizes) - 1):

            self.layers.append(Dense(sizes[i], sizes[i+1], weight_init))

            if i < len(sizes) - 2:
                if activation == "relu":
                    self.activations.append(ReLU())
                elif activation == "sigmoid":
                    self.activations.append(Sigmoid())
                elif activation == "tanh":
                    self.activations.append(Tanh())

        if cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = MeanSquaredError()

        self.lr = cli_args.learning_rate


    def forward(self, X):

        out = X

        for i in range(len(self.layers)):

            out = self.layers[i].forward(out)

            if i < len(self.activations):
                out = self.activations[i].forward(out)

        return out


    def backward(self, y_true, logits):

        if isinstance(self.loss_fn, CrossEntropyLoss):
            grad = self.loss_fn.backward(y_true)
        else:
            grad = self.loss_fn.backward(logits, y_true)

        for i in reversed(range(len(self.layers))):

            # apply activation gradient FIRST (only for hidden layers)
            if i < len(self.activations):
                grad = self.activations[i].backward(grad)

            # then propagate through dense layer
            grad = self.layers[i].backward(grad)

        return grad


    def update_weights(self):

        for layer in self.layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


    def train(self, X_train, y_train, epochs=1, batch_size=32):

        n = X_train.shape[0]

        for epoch in range(epochs):

            indices = np.random.permutation(n)

            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, n, batch_size):

                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                logits = self.forward(X_batch)

                loss = self.loss_fn.forward(logits, y_batch)

                self.backward(y_batch, logits)

                self.update_weights()

            print(f"Epoch {epoch+1} Loss: {loss}")


    def evaluate(self, X, y):

        logits = self.forward(X)

        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)

        accuracy = np.mean(preds == labels)

        return accuracy


    def get_weights(self):

        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()

        return d


    def set_weights(self, weight_dict):

        for i, layer in enumerate(self.layers):

            w_key = f"W{i}"
            b_key = f"b{i}"

            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()

            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()