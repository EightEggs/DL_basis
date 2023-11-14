import numpy as np
import activation_func.activation_func as af
from numpy.random import rand


def init_network() -> dict:
    network = {'W1': rand(2, 3), 'b1': rand(1, 3),
               'W2': rand(3, 2), 'b2': rand(1, 2),
               'W3': rand(2, 2), 'b3': rand(1, 2)}

    return network


def forward(network: dict, x: np.ndarray) -> np.ndarray:
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = af.relu(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = af.relu(a2)
    a3 = np.dot(z2, W3) + b3

    y = af.softmax(a3)
    return y


if __name__ == '__main__':
    network = init_network()
    x = np.array([0, 1])
    y = forward(network, x)
    print(y)
