from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

import activation_func as af


def show_func(func: Callable):
    x = np.arange(-10, 10, 0.1)
    y = func(x)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    show_func(af.step_func)
    show_func(af.sigmoid)
    show_func(af.relu)
    show_func(af.softmax)