import numpy as np


def step_func(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    c = np.max(x)
    exp = np.exp(x - c)  # avoid overflow
    sum_exp = np.sum(exp)
    return exp / sum_exp
