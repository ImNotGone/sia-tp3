from abc import ABC, abstractmethod
import numpy as np


class Perceptron(ABC):
    def __init__(self, input, expected, weights, learn_rate):
        self.input = np.array([[1] + input[i] for i in range(len(input))])
        self.expected = np.array(expected)
        self.weights = np.array(weights)
        self.learn_rate = learn_rate

    def excitement(self, mu):
        return np.dot(self.input[mu], self.weights)

    @abstractmethod
    def activation(self, excitement):
        pass

    @abstractmethod
    def weights_update(self, activation, mu):
        pass

    @abstractmethod
    def error(self):
        pass
