from collections.abc import Callable
from typing import Tuple
import numpy as np
from numpy._typing import NDArray

# ----- Logistic activation function -----


def logistic(x: NDArray, beta: float) -> NDArray:
    return 1 / (1 + np.exp(-2 * beta * x))


def logistic_derivative(x: NDArray, beta: float) -> NDArray:
    return 2 * beta * logistic(x, beta) * (1 - logistic(x, beta))


# ----- Hyperbolic tangent activation function -----


def tanh(x: NDArray, beta: float) -> NDArray:
    return np.tanh(beta * x)


def tanh_derivative(x: NDArray, beta: float) -> NDArray:
    return beta * (1 - np.power(tanh(x, beta), 2))


# ----- relu activation function -----


def relu(x: NDArray) -> NDArray:
    return np.maximum(0, x)


def relu_derivative(x: NDArray) -> NDArray:
    return np.where(x > 0, 1, 0)


# ----- activation function generator -----
ActivationFunction = Callable[[NDArray], NDArray]


def get_activation_function(config) ->  Tuple[ActivationFunction, ActivationFunction]:
    beta = config["beta"]

    if config["activation_function"] == "logistic":
        return lambda x: logistic(x, beta), lambda x: logistic_derivative(x, beta)
    elif config["activation_function"] == "tanh":
        return lambda x: tanh(x, beta), lambda x: tanh_derivative(x, beta)
    elif config["activation_function"] == "relu":
        return relu, relu_derivative
    else:
        raise Exception("Activation function not found")
