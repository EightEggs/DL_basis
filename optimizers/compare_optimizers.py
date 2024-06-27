import numpy as np
import matplotlib.pyplot as plt
from optimizer import SGD, Momentum, AdaGrad, Adam


def f(x, y):
    x = np.array(x)
    y = np.array(y)
    return x**2 + 20 * y**2 + 3 * x * y - 10 * x - 10 * y + 5


def df(x, y):
    x = np.array(x)
    y = np.array(y)
    return 2 * x + 3 * y - 10, 40 * y + 3 * x - 10


init_pos = np.array([-5, 5])
params = {}
params["x"], params["y"] = init_pos[0], init_pos[1]
grads = {}
grads["x"], grads["y"] = 0, 0

SGD = SGD(lr=0.01)
Momentum = Momentum(lr=0.01)
AdaGrad = AdaGrad(lr=0.01)
Adam = Adam(lr=0.01)
optimizers = [SGD, Momentum, AdaGrad, Adam]

fig = plt.figure(1)
x = np.arange(-10, 10, 0.25)
y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

for optimizer in optimizers:
    xs = []
    ys = []
    params["x"], params["y"] = init_pos[0], init_pos[1]

    for i in range(2000):
        xs.append(params["x"])
        ys.append(params["y"])
        grads["x"], grads["y"] = df(params["x"], params["y"])
        optimizer.update(params, grads)

    idx = optimizers.index(optimizer)
    ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow", alpha=0.5)
    ax.scatter(xs, ys, f(xs, ys), c="r", marker="o")
    ax.set_title(optimizer.__class__.__name__)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

plt.show()
