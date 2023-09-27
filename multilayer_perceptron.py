from typing import List, Tuple
import numpy as np
from numpy._typing import NDArray

from activation_functions import ActivationFunction


def multilayer_perceptron(
    data: Tuple[NDArray, NDArray],
    hidden_layer_sizes: List[int],
    output_layer_size: int,
    target_error: float,
    max_epochs: int,
    learning_rate: float,
    batch_size: int,
    neuron_activation_function: ActivationFunction,
    neuron_activation_function_derivative: ActivationFunction,
    optimization_method,
):
    # Initialize weights
    current_network = initialize_weights(
        hidden_layer_sizes, output_layer_size, len(data[0])
    )

    errors_in_epoch = []

    best_error = np.Infinity
    best_network = None
    epoch = 0

    while best_error > target_error and epoch < max_epochs:
        # Get a random training set
        training_set = np.random.choice(data, batch_size)

        # For each training set
        weight_delta: List[NDArray] = []
        error = 0.0

        for input, expected_output in training_set:
            # Propagate the input through the network
            neuron_activations = forward_propagation(
                input, current_network, neuron_activation_function
            )

            # Compute the error
            error += compute_error(neuron_activations[-1], expected_output)

            # Calculate the weight delta
            current_weight_delta = backpropagation(
                neuron_activations,
                expected_output,
                current_network,
                learning_rate,
                neuron_activation_function_derivative,
                optimization_method,
            )

            # Add the weight delta to the total weight delta
            if len(weight_delta) == 0:
                weight_delta = current_weight_delta
            else:
                for i in range(len(weight_delta)):
                    weight_delta[i] += current_weight_delta[i]

        update_weights(current_network, weight_delta)

        errors_in_epoch += [error]

        # If we have a better network, save it
        if error < best_error:
            best_error = error
            best_network = current_network

        epoch += 1

    return best_network, errors_in_epoch


# Initialize weights
# Hidden layer sizes is an array with the number of neurons in each layer
# Output layer is the number of neurons in the output layer
def initialize_weights(
    hidden_layer_sizes: List[int], output_layer_size: int, input_layer_size: int
) -> List[NDArray]:
    # initialize weights
    weights = []

    for i in range(len(hidden_layer_sizes)):
        # Generate random weights for each layer
        # if first layer
        if i == 0:
            weights += [np.random.rand(hidden_layer_sizes[i], input_layer_size)]
        else:
            weights += [
                np.random.rand(hidden_layer_sizes[i], hidden_layer_sizes[i - 1])
            ]

    # add output layer
    weights += [np.random.rand(output_layer_size, hidden_layer_sizes[-1])]

    return weights


def compute_error(output: NDArray, expected_output: NDArray) -> float:
    return np.sum(np.power(output - expected_output, 2)) / 2


def update_weights(weights: List[NDArray], weight_delta: List[NDArray]):
    for i in range(len(weights)):
        weights[i] += weight_delta[i]


def forward_propagation(
    input: NDArray,
    weights: List[NDArray],
    neuron_activation_function: ActivationFunction,
) -> List[NDArray]:
    # Propagate the input through the network
    neuron_activations = []

    # Propagate the input through the network
    previous_layer_output = input
    for i in range(len(weights)):
        weighted_sum = np.dot(weights[i], previous_layer_output)

        activation = neuron_activation_function(weighted_sum)

        neuron_activations += [activation]

        previous_layer_output = activation

    return neuron_activations
