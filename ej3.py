import json
from activation_functions import get_activation_function
from dataset_loaders import get_dataset
from multilayer_perceptron import multilayer_perceptron
from optimization_methods import get_optimization_method


def main():
    config_file = "config.json"

    with open(config_file) as json_file:
        config = json.load(json_file)["ej3"]

        dataset = get_dataset(config)

        hidden_layer_sizes = config["hidden_layers"]
        output_layer_size = len(dataset[0][1])

        target_error = config["target_error"]
        max_epochs = config["max_epochs"]

        batch_size = get_batch_size(config, len(dataset))

        activation_function, activation_function_derivative = get_activation_function(
            config["activation"]
        )

        optimization_method = get_optimization_method(config["optimization"])

        best_network, errors_in_epoch = multilayer_perceptron(
            dataset,
            hidden_layer_sizes,
            output_layer_size,
            target_error,
            max_epochs,
            batch_size,
            activation_function,
            activation_function_derivative,
            optimization_method,
        )



def get_batch_size(config, dataset_size) -> int:
    training_strategy = config["training_strategy"]

    if training_strategy == "batch":
        return dataset_size
    elif training_strategy == "mini_batch":
        batch_size = config["batch_size"]

        if batch_size > dataset_size:
            raise Exception("Batch size is bigger than dataset size")

        return batch_size
    elif training_strategy == "online":
        return 1
    else:
        raise Exception("Training strategy not found")

    
if __name__ == "__main__":
    main()
