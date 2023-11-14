import numpy as np
from decorator import decorator


@decorator
def info(func, *args):
    print(f"{func.__name__}{args[0], args[1]} = {func(*args)}")
    return func(*args)


@info
def AND(x1: float, x2: float) -> int:
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.75
    tmp = np.sum(w * x) + b
    return int(tmp > 0)


@info
def NAND(x1: float, x2: float) -> int:
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.75
    tmp = np.sum(w * x) + b
    return int(tmp > 0)


@info
def OR(x1: float, x2: float) -> int:
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.25
    tmp = np.sum(w * x) + b
    return int(tmp > 0)


@info
def XOR(x1: float, x2: float) -> int:
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


if __name__ == '__main__':
    AND(0, 0)
    AND(1, 0)
    AND(0, 1)
    AND(1, 1)

    NAND(0, 0)
    NAND(1, 0)
    NAND(0, 1)
    NAND(1, 1)

    OR(0, 0)
    OR(1, 0)
    OR(0, 1)
    OR(1, 1)

    XOR(0, 0)
    XOR(1, 0)
    XOR(0, 1)
    XOR(1, 1)
