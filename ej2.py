import random
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from abstract_perceptron import Perceptron
import csv

from activation_functions import get_activation_function

m_values = []
b_values = []
c_values = []


def initialize_weights():
    return [random.uniform(-1.0, 1.0) for _ in range(0, 4)]

class FunctionPerceptron(Perceptron):
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


def learn(input, expected, weights, learn_rate, epsilon, limit, act_func, act_func_der, act_func_norm):
    i = 0
    min_error = sys.maxsize
    perceptron = FunctionPerceptron(input, expected, weights, learn_rate, act_func, act_func_der, act_func_norm)
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
            if i % 100 == 0:
                m_values.append(-min_weights[1] / min_weights[3])
                b_values.append(-min_weights[0] / min_weights[3])
                c_values.append(-min_weights[2] / min_weights[3])

        print(perceptron.weights, min_error)
        i += 1

    for _ in range(20):
        m_values.append(-min_weights[1] / min_weights[3])
        b_values.append(-min_weights[0] / min_weights[3])
        c_values.append(-min_weights[2] / min_weights[3])
    return min_weights


with open("config.json") as config_file:
    config = json.load(config_file)
    file_name = config["ej2"]["data_path"]
    beta = config["ej2"]["beta"]
    epsilon = config["ej2"]["epsilon"]
    learn_rate = config["learning_rate"]
    limit = config["iteration_limit"]
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
learn(input, output, weights, learn_rate, epsilon, limit, act_func, act_func_der, act_func_norm)

# Initialize the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define your arrays of "m," "c," and "b" values (modify these according to your data)

# Create a grid of x and y values
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Initialize the surface plot
Z = np.zeros_like(X)

# Animation function to update the surface plot
def animate(frame):
    m = m_values[frame]
    c = c_values[frame]
    b = b_values[frame]
    ax.clear()
    Z = m * X + c * Y + b
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title(f'z = {m:.2f}x + {c:.2f}y + {b:.2f}')
    # Define the points
    points = np.array(input)
    # Create scatter plots for the points with colors
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', marker='o', s=20, alpha=0.7, edgecolors='k')
    ax.set_zlim(-20, 20)
    ax.view_init(elev=0, azim=5*frame)
    return ax,

# Set up the animation
num_frames = len(m_values)
ani = FuncAnimation(fig, animate, frames=num_frames, interval=200)

# Save the animation as a GIF
ani.save('animated_surface_plot.gif', writer='pillow', fps=5)

plt.show()

