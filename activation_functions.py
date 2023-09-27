import numpy as np

# ----- Logistic activation function -----

def logistic(x, beta):
    return 1 / (1 + np.exp(-2 * beta * x))

def logistic_derivative(x, beta):
    return 2 * beta * logistic(x, beta) * (1 - logistic(x, beta))


# ----- Hyperbolic tangent activation function -----

def tanh(x, beta):
    return np.tanh(beta * x)

def tanh_derivative(x, beta):
    return beta * (1 - np.power(tanh(x, beta), 2))

# ----- relu activation function -----

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# ----- activation function generator -----
def get_activation_function(config):

    beta = config['beta']

    if config['activation_function'] == 'logistic':
        return lambda x: logistic(x, beta), lambda x: logistic_derivative(x, beta)
    elif config['activation_function'] == 'tanh':
        return lambda x: tanh(x, beta), lambda x: tanh_derivative(x, beta)
    elif config['activation_function'] == 'relu':
        return relu, relu_derivative
    else:
        raise Exception('Activation function not found')
