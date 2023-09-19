import random
import numpy as np

def initialize_weights():
    return [random.uniform(-1.0, 1.0) for _ in range(0, 3)]

x = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
y = [-1, -1, -1, 1]

class Perceptron:

    def __init__(self, input, expected, weights, learn_rate):
        self.input = np.array([[1] + input[i] for i in range(len(input))])
        self.expected = np.array(expected)
        self.weights = np.array(weights)
        self.learn_rate = learn_rate

    def exitement(self, mu):
        return np.dot(self.input[mu], self.weights)

    # TODO: use ternary
    def activation(self, exitement):
        if(exitement > 0):
            return 1
        return -1

    def weights_update(self, activation, mu):
        self.weights += (self.learn_rate *(self.expected[mu] - activation) * self.input[mu])
        return self.weights

    # TODO: se puede comprimir?
    def error(self):
        count = 0
        for mu in range(len(self.input)):
            if(self.activation(self.exitement(mu)) != self.expected[mu]):
                count+=1
        return count / len(self.input)

def learn():
    i = 0
    limit = 100
    min_error = 1.0
    epsilon = 0.01
    perceptron = Perceptron(x, y, initialize_weights(), 0.1)
    while (min_error < epsilon and i < limit):
        # get random mu
        mu = random.randint(1, len(x) - 1)

        # compute exitement
        exitement = perceptron.exitement(mu)

        # compute activation
        activation = perceptron.activation(exitement)

        # update_weights
        perceptron.weights_update(activation, mu)

        # compute error
        error = perceptron.error()
        if error < min_error:
            min_error = error
        i += 1
    return

x = [[4.7125, 2.8166]]
y = [-1]
weights = [-1.86, 3.25, 4.3]
perceptron = Perceptron(x, y, weights, 0.1)
print(x, y)
print(perceptron.input)
print(perceptron.expected)
print(perceptron.learn_rate)
print(perceptron.weights)
print(perceptron.exitement(0))
print(perceptron.weights_update(perceptron.activation(perceptron.exitement(0)), 0))
print(perceptron.exitement(0))
