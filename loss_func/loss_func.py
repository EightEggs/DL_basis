import numpy as np


def mean_squared_error(y: np.ndarray, t: np.ndarray) -> float:
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
