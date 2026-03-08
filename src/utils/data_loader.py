"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist


def load_data(dataset="mnist"):
    """
    Load and preprocess dataset using Keras.

    dataset: The name of the dataset to load (e.g., 'mnist' or 'fashion_mnist').
    """
    print("Loading dataset using Keras...")

    if dataset == "mnist":
        (X_train, y_train), (X_val, y_val) = mnist.load_data()
    elif dataset == "fashion_mnist":
        (X_train, y_train), (X_val, y_val) = fashion_mnist.load_data()
    else:
        raise ValueError("Unsupported dataset")

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)

    return X_train, y_train, X_val, y_val