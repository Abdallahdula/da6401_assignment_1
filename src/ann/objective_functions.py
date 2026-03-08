"""
Loss Functions
"""

import numpy as np


class CrossEntropyLoss:

    def forward(self, logits, y_true):

        self.y_true = y_true

        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        loss = -np.sum(y_true * np.log(self.probs + 1e-9)) / logits.shape[0]

        return loss


    def backward(self):

        batch_size = self.y_true.shape[0]

        grad = (self.probs - self.y_true) / batch_size

        return grad


class MeanSquaredError:

    def forward(self, logits, y_true):

        self.logits = logits
        self.y_true = y_true

        loss = np.mean((logits - y_true) ** 2)

        return loss


    def backward(self):

        grad = 2 * (self.logits - self.y_true) / self.y_true.shape[0]

        return grad


def get_loss(name):

    if name == "cross_entropy":
        return CrossEntropyLoss()

    elif name == "mse":
        return MeanSquaredError()

    else:
        raise ValueError("Unknown loss function")
        