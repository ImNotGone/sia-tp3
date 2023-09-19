import random
import numpy as np

def initialize_weights():
    return [random.uniform(-1.0, 1.0) for _ in range(0, 3)]

x = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
y = [-1, -1, -1, 1]

class Perceptron:

    def __init__(self, input, expected):
        self.input = np.array([input[i] + [1] for i in range(len(input))])
        self.expected = np.array(expected)
        self.weigths = np.array(self._initialize_weights())

    def _initialize_weights(self):
        return [random.uniform(-1.0, 1.0) for _ in range(0, 3)]

    def exitement(self, mu):
        return np.dot(self.input[mu], self.weigths)

    # TODO: use ternary
    def activation(self, exitement):
        if(exitement > 0):
            return 1
        return -1

    # TODO: learning rate?
    def weights_update(self, activation, mu):
        self.weigths += ((self.expected[mu] - activation) * self.input[mu])
        return self.weigths

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
    perceptron = Perceptron(x, y)
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

perceptron = Perceptron(x, y)
print(x, y)
print(perceptron.input)
print(perceptron.weigths)
print(perceptron.exitement(0))
