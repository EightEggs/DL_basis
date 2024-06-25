import numpy as np
from two_layer_net_fast import TwoLayerNetFast
from mnist.mnist import load_mnist

# Load the MNIST dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# Define the network
net = TwoLayerNetFast(input_size=784, hidden_size=50, output_size=10)


# Train the network
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 200
learning_rate = 0.12


train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Forward
    loss = net.loss(x_batch, t_batch)

    # Backward
    grad = net.gradient(x_batch, t_batch)
    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * grad[key]

    # Calculate accuracy
    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(loss)
        print(f"train acc = {train_acc}, test acc = {test_acc}, loss = {loss}")


# Plot the loss and accuracy
import matplotlib.pyplot as plt

x = np.arange(len(train_acc_list))
plt.figure(0)
plt.plot(x, train_acc_list, label='train acc', marker='o')
plt.plot(x, test_acc_list, label='test acc', marker='s')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')

plt.figure(1)
plt.plot(x, train_loss_list, label='train loss', marker='o')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc='upper right')

plt.show()

