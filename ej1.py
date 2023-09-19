import random
import numpy as np

def initialize_weights():
    return [random.uniform(-1.0, 1.0) for _ in range(0, 3)]

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
                count += 1
        return count / len(self.input)

def learn(input, expected, weights, learn_rate):
    i = 0
    limit = 10000
    min_error = 1.0
    epsilon = 0.01
    perceptron = Perceptron(input, expected, weights, learn_rate)
    while (min_error > epsilon and i < limit):
        # get random mu
        mu = random.randint(1, len(input) - 1)

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
        print(perceptron.weights, min_error)
        i += 1
    print(perceptron.weights, min_error)
    return

input = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
expected = [-1, -1, -1, 1]
weights = initialize_weights()
learn_rate = 0.1
learn(input, expected, weights, learn_rate)

input = [[4.7125, 2.8166]]
expected = [-1]
weights = [-1.86, 3.25, 4.3]
perceptron = Perceptron(input, expected, weights, learn_rate)
print(input, expected)
print(perceptron.input)
print(perceptron.expected)
print(perceptron.learn_rate)
print(perceptron.weights)
print(perceptron.exitement(0))
print(perceptron.weights_update(perceptron.activation(perceptron.exitement(0)), 0))
print(perceptron.exitement(0))

