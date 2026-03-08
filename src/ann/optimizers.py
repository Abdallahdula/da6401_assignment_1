"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
"""

import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):  # learning_rate: Step size for weight updates.
        self.lr = learning_rate

    def update(self, layer):
        if hasattr(layer, "W"):
            layer.W -= self.lr * layer.grad_W
        if hasattr(layer, "b"):
            layer.b -= self.lr * layer.grad_b


class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.v_w = {}
        self.v_b = {}

    def update(self, layer):

        if id(layer) not in self.v_w:
            self.v_w[id(layer)] = np.zeros_like(layer.W)
            self.v_b[id(layer)] = np.zeros_like(layer.b)

        self.v_w[id(layer)] = self.beta * self.v_w[id(layer)] + (1 - self.beta) * layer.grad_W
        self.v_b[id(layer)] = self.beta * self.v_b[id(layer)] + (1 - self.beta) * layer.grad_b

        layer.W -= self.lr * self.v_w[id(layer)]
        layer.b -= self.lr * self.v_b[id(layer)]


class NAG:

    def __init__(self, learning_rate=0.01, beta=0.9):

        self.lr = learning_rate
        self.beta = beta
        self.v_w = {}
        self.v_b = {}

    def update(self, layer):

        if id(layer) not in self.v_w:
            self.v_w[id(layer)] = np.zeros_like(layer.W)
            self.v_b[id(layer)] = np.zeros_like(layer.b)

        v_prev_w = self.v_w[id(layer)]
        v_prev_b = self.v_b[id(layer)]

        self.v_w[id(layer)] = self.beta * self.v_w[id(layer)] + self.lr * layer.grad_W
        self.v_b[id(layer)] = self.beta * self.v_b[id(layer)] + self.lr * layer.grad_b

        layer.W -= (-self.beta * v_prev_w + (1 + self.beta) * self.v_w[id(layer)])
        layer.b -= (-self.beta * v_prev_b + (1 + self.beta) * self.v_b[id(layer)])


class RMSProp:

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):

        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s_w = {}
        self.s_b = {}

    def update(self, layer):

        if id(layer) not in self.s_w:
            self.s_w[id(layer)] = np.zeros_like(layer.W)
            self.s_b[id(layer)] = np.zeros_like(layer.b)

        self.s_w[id(layer)] = self.beta * self.s_w[id(layer)] + (1 - self.beta) * (layer.grad_W ** 2)
        self.s_b[id(layer)] = self.beta * self.s_b[id(layer)] + (1 - self.beta) * (layer.grad_b ** 2)

        layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_w[id(layer)]) + self.epsilon)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[id(layer)]) + self.epsilon)


class Adam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_w = {}
        self.m_b = {}
        self.v_w = {}
        self.v_b = {}
        self.t = 0

    def update(self, layer):

        if id(layer) not in self.m_w:
            self.m_w[id(layer)] = np.zeros_like(layer.W)
            self.m_b[id(layer)] = np.zeros_like(layer.b)
            self.v_w[id(layer)] = np.zeros_like(layer.W)
            self.v_b[id(layer)] = np.zeros_like(layer.b)

        self.t += 1

        self.m_w[id(layer)] = self.beta1 * self.m_w[id(layer)] + (1 - self.beta1) * layer.grad_W
        self.m_b[id(layer)] = self.beta1 * self.m_b[id(layer)] + (1 - self.beta1) * layer.grad_b

        self.v_w[id(layer)] = self.beta2 * self.v_w[id(layer)] + (1 - self.beta2) * (layer.grad_W ** 2)
        self.v_b[id(layer)] = self.beta2 * self.v_b[id(layer)] + (1 - self.beta2) * (layer.grad_b ** 2)

        m_w_hat = self.m_w[id(layer)] / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b[id(layer)] / (1 - self.beta1 ** self.t)

        v_w_hat = self.v_w[id(layer)] / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b[id(layer)] / (1 - self.beta2 ** self.t)

        layer.W -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


class Nadam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_w = {}
        self.m_b = {}
        self.v_w = {}
        self.v_b = {}
        self.t = 0

    def update(self, layer):

        if id(layer) not in self.m_w:
            self.m_w[id(layer)] = np.zeros_like(layer.W)
            self.m_b[id(layer)] = np.zeros_like(layer.b)
            self.v_w[id(layer)] = np.zeros_like(layer.W)
            self.v_b[id(layer)] = np.zeros_like(layer.b)

        self.t += 1

        self.m_w[id(layer)] = self.beta1 * self.m_w[id(layer)] + (1 - self.beta1) * layer.grad_W
        self.m_b[id(layer)] = self.beta1 * self.m_b[id(layer)] + (1 - self.beta1) * layer.grad_b

        self.v_w[id(layer)] = self.beta2 * self.v_w[id(layer)] + (1 - self.beta2) * (layer.grad_W ** 2)
        self.v_b[id(layer)] = self.beta2 * self.v_b[id(layer)] + (1 - self.beta2) * (layer.grad_b ** 2)

        m_w_hat = self.m_w[id(layer)] / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b[id(layer)] / (1 - self.beta1 ** self.t)

        v_w_hat = self.v_w[id(layer)] / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b[id(layer)] / (1 - self.beta2 ** self.t)

        nesterov_w = self.beta1 * m_w_hat + ((1 - self.beta1) * layer.grad_W) / (1 - self.beta1 ** self.t)
        nesterov_b = self.beta1 * m_b_hat + ((1 - self.beta1) * layer.grad_b) / (1 - self.beta1 ** self.t)

        layer.W -= self.lr * nesterov_w / (np.sqrt(v_w_hat) + self.epsilon)
        layer.b -= self.lr * nesterov_b / (np.sqrt(v_b_hat) + self.epsilon)



def get_optimizer(name, learning_rate):

    name = name.lower()

    if name == "sgd":
        return SGD(learning_rate)

    elif name == "momentum":
        return Momentum(learning_rate)

    elif name == "nag":
        return NAG(learning_rate)

    elif name == "rmsprop":
        return RMSProp(learning_rate)

    elif name == "adam":
        return Adam(learning_rate)

    elif name == "nadam":
        return Nadam(learning_rate)

    else:
        raise ValueError(f"Unknown optimizer: {name}")