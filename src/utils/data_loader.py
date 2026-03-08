from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
import numpy as np


def one_hot(y, num_classes=10):

    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded


def load_data(dataset):

    if dataset == "mnist":
        (X_train, y_train), _ = mnist.load_data()

    elif dataset == "fashion_mnist":
        (X_train, y_train), _ = fashion_mnist.load_data()

    X_train = X_train.reshape(-1, 784) / 255.0

    y_train = one_hot(y_train)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=42
    )

    return X_train, y_train, X_val, y_val