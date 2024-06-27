import numpy as np


def step_func(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)  # 避免溢出
    y = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    return y
