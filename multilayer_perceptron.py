import numpy as np


def multilayer_perceptron(
    data,
    hidden_layer_sizes,
    output_layer_size,
    target_error,
    max_epochs,
    learning_rate,
    batch_size,
    neuron_activation_function,
    neuron_activation_function_derivative,
    descent_direction_function,
):
    # Initialize weights
    current_network = initialize_weights(
        hidden_layer_sizes, output_layer_size, len(data[0])
    )

    errors_in_epoch = []

    error = None
    best_error = np.Infinity
    best_network = None
    epoch = 0

    while best_error > target_error and epoch < max_epochs:
        # Get a random training set
        training_set = np.random.choice(data, batch_size)

        # For each training set
        weight_delta = []
        error = 0

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
                descent_direction_function,
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
def initialize_weights(hidden_layer_sizes, output_layer_size, input_layer_size):
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


def compute_error(output, expected_output):
    return np.sum(np.power(output - expected_output, 2)) / 2

def update_weights(weights, weight_delta):
    for i in range(len(weights)):
        weights[i] += weight_delta[i]

def forward_propagation(input, weights, neuron_activation_function):
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
    
