"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np


class Dense:
    def __init__(self, input_size, output_size, weight_init="random"):
        if weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
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
        m = self.input.shape[0]
       
        self.grad_W = np.dot(self.input.T, dZ) / m
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / m

        dX = np.dot(dZ, self.W.T)

        return dX
