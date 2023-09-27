from typing import List, Tuple
import numpy as np
from numpy._typing import NDArray
import  random

from activation_functions import ActivationFunction
from optimization_methods import OptimizationMethod


def multilayer_perceptron(
    data: List[Tuple[NDArray, NDArray]],
    hidden_layer_sizes: List[int],
    output_layer_size: int,
    target_error: float,
    max_epochs: int,
    batch_size: int,
    neuron_activation_function: ActivationFunction,
    neuron_activation_function_derivative: ActivationFunction,
    optimization_method: OptimizationMethod,
):
    # Initialize weights
    current_network = initialize_weights(
        hidden_layer_sizes, output_layer_size, len(data[0][0])
    )

    errors_in_epoch = []

    best_error = np.Infinity
    best_network = None
    epoch = 0

    while best_error > target_error and epoch < max_epochs:
        # Get a random training set
        training_set = random.sample(data, batch_size)

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
                neuron_activation_function_derivative,
                input,
            )

            # Add the weight delta to the total weight delta
            if len(weight_delta) == 0:
                weight_delta = current_weight_delta
            else:
                for i in range(len(weight_delta)):
                    weight_delta[i] += current_weight_delta[i]

        # Average the weight delta and apply the optimization method
        weight_delta = [delta / batch_size for delta in weight_delta]
        weight_delta = optimization_method(weight_delta)

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

# Uso de diccionarios para guardar los deltas. Ver si esta bien.
# Me parece que Weight_deltas se guarda desordenado
def backpropagation(
    neuron_activations: List[NDArray],
    expected_output: NDArray,
    network: List[NDArray],
    neuron_activation_function_derivative: ActivationFunction,
    input: NDArray,
) -> List[NDArray]:
    deltas = list()
    weight_deltas = []
    for layer_idx in reversed(range(len(network))):
        layer = network[layer_idx]
        if layer_idx == len(network)-1:
            for neuron_idx in range(len(layer)):
                neuron = layer[neuron_idx]
                #Para calcular error de output
                neuron_activation=neuron_activations[-1][neuron_idx]
                
                #Calculo h
                prev_layer_output=neuron_activations[-2]
                h=np.dot(layer[neuron_idx], prev_layer_output)
                # ( Expected - Actual ) * derivada
                # TODO chequear que estoy accediendo bien. 
                delta = (expected_output[neuron_idx]-neuron_activation) * neuron_activation_function_derivative(h)
                deltas.append(delta)
                # TODO ver si es correcto guardarse asi los deltas para utilizarlos dsp
                neuron['delta'] = delta
                # La idea es guardar en weight_deltas lo que seria el deltaw SIN learning rate
                # en prev_layer_output esta el vector de activaciones, y lo multiplico por el delta
                weight_deltas += [delta * prev_layer_output]
        else:
            for neuron_idx_curr_layer in range(len(layer)):
                
                error=0.0
            
                # Sumatoria de deltas * weight respectivo hacia curr_neuron
                for neuron_superior_layer in network[layer_idx+1]:
                    #TODO chequear que estoy accediendo bien
                    error+= neuron_superior_layer['delta'] * network[layer_idx+1][neuron_idx_curr_layer]
                    
                neuron=layer[neuron_idx_curr_layer]
                ###neuron_activation=neuron_activations[layer_idx][neuron_idx]
                #Calculo h. TODO ver si es correcto y si se puede pasar directamente weightedsums de forward
                if(layer_idx==0):
                    prev_layer_output=input
                else:
                    prev_layer_output=neuron_activations[layer_idx-1]
                h=np.dot(layer[neuron_idx_curr_layer], prev_layer_output)
                
                #Sumatoria * derivada
                delta = error * neuron_activation_function_derivative(h)
                neuron['delta'] = delta
                deltas.append(delta)
                weight_deltas += [delta * prev_layer_output]
    return weight_deltas




