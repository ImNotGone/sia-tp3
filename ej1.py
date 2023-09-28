import copy
import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from abstract_perceptron import Perceptron

# Define your matrix of "m" and "b" values (modify this according to your data)
m_values = []
b_values = []


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
            min_weights = copy.copy(perceptron.weights)
            m_values.append(-min_weights[1] / min_weights[2])
            b_values.append(-min_weights[0] / min_weights[2])

        print(perceptron.weights, min_error)
        i += 1

    for _ in range(4):
        m_values.append(-min_weights[1] / min_weights[2])
        b_values.append(-min_weights[0] / min_weights[2])
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


# Function to calculate the linear equation y = mx + b
def linear_function(x, m, b):
    return m * x + b


# Initialize plot and line objects
fig, ax = plt.subplots()
x_data = np.linspace(-1.5, 1.5, 200)
line, = ax.plot(x_data, linear_function(x_data, 0, 0))

# Define the points
points = np.array(input)

# Define the color array based on expected
colors = ['red' if val == -1 else 'green' for val in expected]

# Create scatter plots for the points with colors
ax.scatter(points[:, 0], points[:, 1], c=colors, marker='o')


# Animation function to update the line
def animate(frame):
    m = m_values[frame]
    b = b_values[frame]
    line.set_ydata(linear_function(x_data, m, b))
    ax.set_title(f'y = {m:.2f}x + {b:.2f}')
    return line,


# Set up the animation
num_frames = len(m_values)
ani = FuncAnimation(fig, animate, frames=num_frames, interval=500)

# Display the animation
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.ylim(-5, 5)  # Adjust the y-axis limits if needed
plt.xlim(-1.5, 1.5)  # Adjust the x-axis limits if needed
# Save the animation as a GIF
ani.save('animated_plot.gif', writer='pillow')
plt.show()
