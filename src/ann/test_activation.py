import numpy as np
from activations import ReLU, Sigmoid, Tanh

X = np.random.randn(5,3)

relu = ReLU()
out = relu.forward(X)
grad = relu.backward(np.ones_like(out))

print("ReLU works:", out.shape, grad.shape)

sig = Sigmoid()
out = sig.forward(X)
grad = sig.backward(np.ones_like(out))

print("Sigmoid works:", out.shape, grad.shape)

tanh = Tanh()
out = tanh.forward(X)
grad = tanh.backward(np.ones_like(out))

print("Tanh works:", out.shape, grad.shape)
