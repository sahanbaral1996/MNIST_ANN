import numpy as np

class Neural:

    def __init__(self, input, output, weights1, weights2,weights3):
        self.input = input
        self.output = output
        self.weights1 = weights1
        self.weights2 = weights2
        self.weights3 = weights3
        self.error = []


    def forward(self):
        self.layer1 = np.dot(self.input, self.weights1)
        self.layer2 = np.dot(self.tanh(self.layer1), self.weights2)
        self.layer3 = np.dot(self.tanh(self.layer2), self.weights3)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self,x):
        return 1 - (np.tanh(x) ** 2)


    def back_propagate(self):
        common_part = 2*(self.output-self.tanh(self.layer3))/self.output.shape[0]

        self.error.append(np.mean(np.power(self.output-self.tanh(self.layer3), 2)))

        d_weights3 = np.dot(self.tanh(self.layer2).T, (common_part* self.tanh_prime(self.layer3)))

        d_weights2 = np.dot(self.layer1.T, (np.dot((common_part*self.tanh_prime(self.layer3)),self.weights3.T)*self.tanh_prime(self.layer2)))

        d_weights1 = np.dot(self.input.T,(np.dot((np.dot((common_part * self.tanh_prime(self.layer3)), self.weights3.T) * self.tanh_prime(self.layer2)),self.weights2.T)*self.tanh_prime(self.layer1)))


        self.weights1 =self.weights1 + (0.1*d_weights1)
        self.weights2 =self.weights2 + (0.1*d_weights2)
        self.weights3 = self.weights3 + (0.1*d_weights3)
