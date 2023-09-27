import random
import sys
import json

from abstract_perceptron import Perceptron
import csv

from activation_functions import get_activation_function


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
    def __init__(self, input, expected, weights, learn_rate, activation_function, activation_function_derivative,
                 activation_function_normalize):
        super().__init__(input, expected, weights, learn_rate)
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.activation_function_normalize = activation_function_normalize

    # we choose which function to use
    def activation(self, excitement):
        return self.activation_function(excitement)

    def error(self):
        partial_err = 0.0
        for mu in range(len(self.input)):
            partial_err += (self.activation_function_normalize(self.expected[mu]) - self.activation(
                self.excitement(mu))) ** 2
        return partial_err / 2

    def weights_update(self, activation, mu):
        self.weights += self.learn_rate * (
                self.activation_function_normalize(
                    self.expected[mu]) - activation) * self.activation_function_derivative(self.excitement(mu)) * \
                        self.input[mu]
        return self.weights


def learn(input, expected, weights, learn_rate, act_func, act_func_der, act_func_norm):
    i = 0
    limit = 100000
    min_error = sys.maxsize
    epsilon = 0.001
    perceptron = NonLinearPerceptron(input, expected, weights, learn_rate, act_func, act_func_der, act_func_norm)
    input_len = len(input)
    min_weights = []

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
            min_weights = perceptron.weights
        print(perceptron.weights, min_error)
        i += 1
    print(perceptron.weights, min_error)
    return min_weights


with open("config.json") as config_file:
    config = json.load(config_file)
    file_name = config["ej2"]["data_path"]
    beta = config["ej2"]["beta"]
    (act_func, act_func_der, act_func_norm) = get_activation_function(config["ej2"]["activation_function"], beta)

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
learn(input, output, weights, learn_rate, act_func, act_func_der, act_func_norm)
