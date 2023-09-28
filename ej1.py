import random
import json
from abstract_perceptron import Perceptron

def initialize_weights():
    return [random.uniform(-1.0, 1.0) for _ in range(0, 3)]


class SimplePerceptron(Perceptron):

    def activation(self, excitement):
        return 1 if excitement > 0 else -1

    def error(self):
        count = 0
        for mu in range(len(self.input)):
            if (self.activation(self.excitement(mu)) != self.expected[mu]):
                count += 1
        return count / len(self.input)
    
    def weights_update(self, activation, mu):
        self.weights += (self.learn_rate * (self.expected[mu] - activation) * self.input[mu])
        return self.weights


def learn(input, expected, weights, learn_rate, limit, min_error):
    i = 0
    perceptron = SimplePerceptron(input, expected, weights, learn_rate)
    input_len = len(input)
    min_weights = []
    while (min_error > 0 and i < limit):
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
    training_set = config["ej1"]["training_set"]
    input = config["ej1"]["input"][training_set]
    expected = config["ej1"]["expected"][training_set]
    min_error = config["ej1"]["min_error"]
    learn_rate = config["learning_rate"]
    limit = config["iteration_limit"]

weights = initialize_weights()
learn(input, expected, weights, learn_rate, limit, min_error)

# input = [[4.7125, 2.8166]]
# expected = [-1]
# weights = [-1.86, 3.25, 4.3]
# perceptron = SimplePerceptron(input, expected, weights, learn_rate)
# print(input, expected)
# print(perceptron.input)
# print(perceptron.expected)
# print(perceptron.learn_rate)
# print(perceptron.weights)
# print(perceptron.excitement(0))
# print(perceptron.weights_update(perceptron.activation(perceptron.excitement(0)), 0))
# print(perceptron.excitement(0))
