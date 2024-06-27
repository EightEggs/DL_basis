from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def numerical_diff(f: Callable, x: float) -> float:
    h = 1e-7
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_grad(f: Callable, x: np.ndarray) -> np.ndarray:
    h = 1e-7
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()   
        
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
