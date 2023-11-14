import numpy as np
import pickle
from mnist.mnist import load_mnist
from activation_func.activation_func import sigmoid, softmax, relu


def get_data():
    (img_train, label_train), (img_test, label_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return img_test, label_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = np.zeros_like(a3)
    y[np.argmax(a3)] = 1

    return y


x, la = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    if np.array_equal(y, la[i]):
        accuracy_cnt += 1

print("Accuracy:" + str(accuracy_cnt / len(x)))
