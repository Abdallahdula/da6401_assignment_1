"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, X):
        self.input = X
        return np.maximum(0, X)

    def backward(self, dA):
        dX = dA * (self.input > 0)
        return dX


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, dA):
        dX = dA * (self.output * (1 - self.output))
        return dX


class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, X):
        self.output = np.tanh(X)
        return self.output

    def backward(self, dA):
        dX = dA * (1 - self.output ** 2)
        return dX
