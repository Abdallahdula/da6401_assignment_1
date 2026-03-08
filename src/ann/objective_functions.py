"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np


class MeanSquaredError:

    def forward(self, y_pred, y_true):
        # y_pred: Predicted values from the model.
        # y_true: Ground truth labels.
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self, y_pred, y_true):
        m = y_true.shape[0]
        grad = 2 * (y_pred - y_true) / m
        return grad



class CrossEntropyLoss:

    def softmax(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, logits, y_true):
        self.probs = self.softmax(logits)
        m = y_true.shape[0]

        loss = -np.sum(y_true * np.log(self.probs + 1e-9)) / m

        return loss


    def backward(self, y_true):
        m = y_true.shape[0]

        grad = (self.probs - y_true) / m

        return grad

def get_loss(name):

    name = name.lower()

    if name in ["cross_entropy", "crossentropy", "ce"]:
        return CrossEntropyLoss()

    elif name in ["mse", "mean_squared_error"]:
        return MeanSquaredError()

    else:
        raise ValueError(f"Unknown loss function: {name}")