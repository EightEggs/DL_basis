from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def numerical_diff(f: Callable, x: float) -> float:
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_grad(f: Callable, x: np.ndarray) -> np.ndarray:
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.shape[0]):
        x0 = x[i]
        x[i] = x0 + h
        fx1 = f(x)

        x[i] = x0 - h
        fx2 = f(x)

        grad[i] = (fx1 - fx2) / (2 * h)
        x[i] = x0

    return grad


def grad_desc(f: Callable, init_x: np.ndarray, lr: float = 0.01, steps: int = 100) -> ndarray:
    x = init_x
    xs = []
    for i in range(steps):
        xs.append(x.copy())
        grad = numerical_grad(f, x)
        x -= lr * grad
    return np.array(xs)


if __name__ == '__main__':
    def test_func(x):
        return x[0] ** 2 + x[1] ** 2 - 1.5 * x[0] * x[1]


    test_x = np.array([-9.0, 7.8])
    test_xs = grad_desc(test_func, test_x, lr=0.06)

    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = test_func((X, Y))

    plt.figure(figsize=(10, 10), dpi=200)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.contourf(X, Y, Z)
    plt.scatter(test_xs[:, 0], test_xs[:, 1], c='r', s=8)
    plt.show()
