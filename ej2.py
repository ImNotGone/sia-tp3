import random
import sys
from abstract_perceptron import Perceptron


def initialize_weights():
    return [random.uniform(-1.0, 1.0) for _ in range(0, 4)]

# loads input / output from csv file
def load_data(file_name):
    input = []
    output = []

    i = 0
    # read from file mats of size rows x N and return them in an array
    with open(file_name, "r") as data:
        for data_row in data:
            # skip first row
            if i == 0:
                i+=1
                continue

            row = []
            pruned_row = data_row.replace('\n', "").split(',')
            for data_col in pruned_row:
                row += [float(data_col)]
            input += [row[:3]]
            output += [row[3]]
    return (input, output)


class LinearPerceptron(Perceptron):

    # In Linear Preceptron, the activation function is the identity
    def activation(self, excitement):
        return excitement

    def error(self):
        partial_err = 0
        for mu in range(len(self.input)):
            partial_err += (self.activation(self.excitement(mu)) - self.expected[mu])** 2
        return 0.5 * partial_err


def learn(input, expected, weights, learn_rate):
    i = 0
    limit = 10000
    min_error = sys.maxsize
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

file_name = "./data/ej2-conjunto.csv"
(input, output) = load_data(file_name)
for i in range(len(input)):
    print(input[i], output[i])
    print()

weights = initialize_weights()
learn_rate = 0.1
learn(input, output, weights, learn_rate)

