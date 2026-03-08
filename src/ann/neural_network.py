"""
Neural Network implementation
Handles forward and backward propagation
"""

import numpy as np
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh


class NeuralNetwork:

    def __init__(self, *args, **kwargs):

        self.layers = []
        self.activations = []

        # --------------------------------
        # Case 1: autograder -> NeuralNetwork(args)
        # --------------------------------
        if len(args) == 1 and not kwargs:

            cli_args = args[0]

            input_size = 784
            output_size = 10
            activation = cli_args.activation
            weight_init = cli_args.weight_init

            if cli_args.hidden_size is not None:
                hidden_sizes = cli_args.hidden_size
            elif cli_args.num_layers is not None:
                hidden_sizes = [cli_args.num_neurons] * cli_args.num_layers
            else:
                hidden_sizes = [cli_args.num_neurons] * cli_args.hidden_layers

        # --------------------------------
        # Case 2: your train.py call
        # --------------------------------
        else:

            input_size = kwargs.get("input_size", args[0] if len(args) > 0 else None)
            hidden_sizes = kwargs.get("hidden_sizes", args[1] if len(args) > 1 else None)
            output_size = kwargs.get("output_size", args[2] if len(args) > 2 else None)
            activation = kwargs.get("activation", "relu")
            weight_init = kwargs.get("weight_init", "xavier")

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Build layers
        for i in range(len(layer_sizes) - 1):

            layer = Dense(layer_sizes[i], layer_sizes[i + 1], weight_init)
            self.layers.append(layer)

            # Add activation for hidden layers
            if i < len(layer_sizes) - 2:

                if activation == "relu":
                    self.activations.append(ReLU())

                elif activation == "sigmoid":
                    self.activations.append(Sigmoid())

                elif activation == "tanh":
                    self.activations.append(Tanh())

                else:
                    raise ValueError("Unsupported activation")

    # --------------------------------

    def forward(self, X):

        output = X

        for i in range(len(self.layers)):

            output = self.layers[i].forward(output)

            if i < len(self.activations):
                output = self.activations[i].forward(output)

        return output

    # --------------------------------

    def backward(self, grad):

        for i in reversed(range(len(self.layers))):

            if i < len(self.activations):
                grad = self.activations[i].backward(grad)

            grad = self.layers[i].backward(grad)

        return grad

    # --------------------------------

    def get_parameters(self):

        params = {}

        for i, layer in enumerate(self.layers):
            params[f"W{i}"] = layer.W
            params[f"b{i}"] = layer.b

        return params

    # --------------------------------

    def set_parameters(self, params):

        for i, layer in enumerate(self.layers):

            layer.W = params[f"W{i}"]
            layer.b = params[f"b{i}"]