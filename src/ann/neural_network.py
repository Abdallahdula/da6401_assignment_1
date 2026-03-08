"""
Main Neural Network Model class
Handles forward and backward propagation
"""

import numpy as np
from .neural_layer import Dense
from .activations import ReLU, Sigmoid, Tanh


class NeuralNetwork:

    def __init__(self, input_size, hidden_sizes, output_size, activation="relu", weight_init="xavier"):

        self.layers = []
        self.activations = []

        sizes = [input_size] + hidden_sizes + [output_size]

        # Build layers
        for i in range(len(sizes) - 1):

            self.layers.append(Dense(sizes[i], sizes[i+1], weight_init))

            if i < len(sizes) - 2:

                if activation == "relu":
                    self.activations.append(ReLU())

                elif activation == "sigmoid":
                    self.activations.append(Sigmoid())

                elif activation == "tanh":
                    self.activations.append(Tanh())

                else:
                    raise ValueError("Unsupported activation")

    # ------------------------------------------------

    def forward(self, X):

        out = X

        for i in range(len(self.layers)):

            out = self.layers[i].forward(out)

            if i < len(self.activations):
                out = self.activations[i].forward(out)

        return out

    # ------------------------------------------------

    def backward(self, grad):

        for i in reversed(range(len(self.layers))):

            if i < len(self.activations):
                grad = self.activations[i].backward(grad)

            grad = self.layers[i].backward(grad)

        return grad

    # ------------------------------------------------

    def get_parameters(self):

        params = {}

        for i, layer in enumerate(self.layers):

            params[f"W{i}"] = layer.W
            params[f"b{i}"] = layer.b

        return params

    # ------------------------------------------------

    def set_parameters(self, params):

        for i, layer in enumerate(self.layers):

            layer.W = params[f"W{i}"]
            layer.b = params[f"b{i}"]