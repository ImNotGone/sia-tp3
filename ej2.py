import random
import sys
from abstract_perceptron import Perceptron
import numpy as np
import csv

BETA = 1

SIGMOID = (lambda x: 1 / (1 + np.exp(-2 * BETA * x)))
SIGMOID_DERIVATIVE = (lambda x: 2 * BETA * SIGMOID(x) * (1 - SIGMOID(x)))

TAN_H = (lambda x: np.tanh(x))
TAN_H_DERIVATIVE = (lambda x: 1 - (np.tanh(x) ** 2))


def initialize_weights():
    return [random.uniform(-1.0, 1.0) for _ in range(0, 4)]

class LinearPerceptron(Perceptron):

    # In Linear Preceptron, the activation function is the identity
    def activation(self, excitement):
        return excitement

    def error(self):
        partial_err = 0.0
        for mu in range(len(self.input)):
            partial_err += (self.expected[mu] - self.activation(self.excitement(mu))) ** 2
        return partial_err / 2

    def weights_update(self, activation, mu):
        self.weights += (self.learn_rate * (self.expected[mu] - activation) * self.input[mu])
        return self.weights


class NonLinearPerceptron(Perceptron):

    # we choose which function to use
    def activation(self, excitement):
        return SIGMOID(excitement)

    def error(self):
        partial_err = 0.0
        for mu in range(len(self.input)):
            partial_err += (self.expected[mu] - self.activation(self.excitement(mu))) ** 2
        return partial_err / 2

    def weights_update(self, activation, mu):
        self.weights += self.learn_rate * (self.expected[mu] - activation) * SIGMOID_DERIVATIVE(self.excitement(mu)) * self.input[mu]
        return self.weights


def learn(input, expected, weights, learn_rate):
    i = 0
    limit = 10000
    min_error = sys.maxsize
    epsilon = 0.001
    perceptron = NonLinearPerceptron(input, expected, weights, learn_rate)
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


file_name = "./data/ej2-conjunto.csv"
input = []
output = []
with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            input += [[float(i) for i in row[:3]]]
            output += [float(row[3])]
            line_count += 1

for i in range(len(input)):
    print(input[i], output[i])
    print()

weights = initialize_weights()
learn_rate = 0.1
learn(input, output, weights, learn_rate)
