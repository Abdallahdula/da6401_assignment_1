"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np


class Dense:
    def __init__(self, input_size, output_size, weight_init="random"):
        # input_size: Number of input features.
        # output_size: Number of output features.
        # weight_init: Method for weight initialization ('xavier', 'he', or 'zeros').

        if weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))

        elif weight_init == "he":
            std = np.sqrt(2 / input_size)
            self.W = np.random.randn(input_size, output_size) * std

        elif weight_init == "zeros":
            self.W = np.zeros((input_size, output_size))

        else:
            self.W = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros((1, output_size))

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.input = None

    def forward(self, X):

        self.input = X
        Z = np.dot(X, self.W) + self.b

        return Z

    def backward(self, dZ):

        # gradient w.r.t weights
        self.grad_W = np.dot(self.input.T, dZ)

        # gradient w.r.t bias (keep correct shape)
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)

        # gradient w.r.t input
        dX = np.dot(dZ, self.W.T)

        return dX