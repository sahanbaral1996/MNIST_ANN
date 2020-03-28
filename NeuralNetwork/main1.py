import numpy as np
import NeuralNetwork.NeuralNety as n
import Network
import FullyConnected
import ActivationLayer
import Function
import Error
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
np.set_printoptions(suppress=True)

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = utils.to_categorical(y_test)

weights1 = np.random.rand(28*28,400)-0.5
weights2 = np.random.rand(400,100)-0.5
weights3 = np.random.rand(100,10)-0.5

n1 = n.Neural(x_train[0:1000], y_train[0:1000], weights1,weights2,weights3)
for i in range(800):
    n1.forward()
    n1.back_propagate()

print(n1.error)

layer1 = np.dot(x_test[0:10], n1.weights1)
layer2 = np.dot(n1.tanh(layer1), n1.weights2)
layer3 = np.dot(n1.tanh(layer2), n1.weights3)
print(n1.tanh(layer3))

print(y_test[0:10])


