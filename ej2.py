import random
from abstract_perceptron import Perceptron


def initialize_weights():
    return [random.uniform(-1.0, 1.0) for _ in range(0, 4)]


class LinearPerceptron(Perceptron):

    # In Linear Preceptron, the activation function is the identity
    def activation(self, excitement):
        return excitement

    def error(self):
        partial_err = 0
        for mu in range(len(self.input)):
            partial_err += pow(self.activation(self.excitement(mu) - self.expected[mu]), 2)
        return 0.5 * partial_err


def learn(input, expected, weights, learn_rate):
    i = 0
    limit = 10000
    min_error = 1.0
    epsilon = 0.01
    perceptron = LinearPerceptron(input, expected, weights, learn_rate)
    input_len = len(input)
    while (min_error > epsilon and i < limit):
        # get random mu
        mu = random.randint(0, input_len - 1)

        # compute excitement
        excitement = perceptron.excitement(mu)

        # compute activation
        activation = perceptron.activation(excitement)

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
