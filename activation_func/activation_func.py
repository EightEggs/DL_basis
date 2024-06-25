import numpy as np


def step_func(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # 避免溢出
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)  # 避免溢出
    return np.exp(x) / np.sum(np.exp(x))

