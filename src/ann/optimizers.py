"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, layer):
        if hasattr(layer, "W"):
            layer.W = layer.W - self.lr * layer.grad_W

        if hasattr(layer, "b"):
            layer.b = layer.b - self.lr * layer.grad_b