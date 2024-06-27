import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from mnist.mnist import load_mnist
from optimizer import SGD, Momentum, AdaGrad, Adam
from two_layer_net.two_layer_net_fast import TwoLayerNetFast

SGD = SGD(lr=0.01)
Momentum = Momentum(lr=0.01)
AdaGrad = AdaGrad(lr=0.01)
Adam = Adam(lr=0.01)
optimizers = {"SGD": SGD, "Momentum": Momentum, "AdaGrad": AdaGrad, "Adam": Adam}

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 3000

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = TwoLayerNetFast(input_size=784, hidden_size=50, output_size=10)
    train_loss[key] = []

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("=============" + "iteration:" + str(i) + "=============")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


def smooth_curve(x):
    window_len = 11
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[5 : len(y) - 5]


markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(
        x,
        smooth_curve(train_loss[key]),
        marker=markers[key],
        markevery=100,
        label=key,
        markersize=3,
        linewidth=1,
    )
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()
