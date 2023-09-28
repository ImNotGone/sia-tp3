import json
import csv
from activation_functions import get_activation_function
from dataset_loaders import get_dataset
from dataset_loaders import create_noise
from dataset_loaders import create_numbers_identifier_dataset_with_noise
from multilayer_perceptron import multilayer_perceptron, predict
from optimization_methods import get_optimization_method
import matplotlib.pyplot as plt


def main():
    config_file = "config.json"
    archivo_csv = "datos4.csv"

    # Abre el archivo CSV y muestra su contenido actual
    contenido_actual = abrir_csv(archivo_csv)
    
    with open(config_file) as json_file:
        config = json.load(json_file)["ej3"]

        dataset = get_dataset(config)
        noise_data = create_numbers_identifier_dataset_with_noise()

        hidden_layer_sizes = config["hidden_layers"]
        output_layer_size = len(dataset[0][1])

        target_error = config["target_error"]
        #max_epochs = config["max_epochs"]

        batch_size = get_batch_size(config, len(dataset))
        activation_function, activation_function_derivative = get_activation_function(
                config["activation"]
            )
        i=0
        while i <300:
            i+=2
            
            max_epochs = i

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

            #for error in errors_in_epoch:
                #print(error)

            
            
            get_accuracy(dataset,best_network,max_epochs,activation_function,archivo_csv)
            get_accuracy(noise_data,best_network,max_epochs,activation_function,archivo_csv)

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

def get_accuracy(
        data,
        best_network,
        epochs,
        activation_function,
        archivo_csv,
    ):
        tp=0
        fp=0
        tn=0
        fn=0
        threshold=0.5
        for input, expected_output in data:
            final_neuron_activations=predict(input,best_network,activation_function)
            max=-1
            max_idx=-1
            #print("gg")
            exp=-1
            for t in range(len(expected_output)):
                if expected_output[t]==1:
                    exp=t
            for i in range(len(final_neuron_activations)):
                #print("gg")
                if final_neuron_activations[i]>=threshold:
                    if exp==i:
                        tp+=1
                    else:
                        fp+=1
                else:
                    if exp==i:
                        fn+=1
                    else:
                        tn+=1
                #if(final_neuron_activations[i]>max):
                    #max=final_neuron_activations[i]
                    #max_idx=i
                #print(final_neuron_activations)
                #print(final_neuron_activations[i])
                #if final_neuron_activations[i]
                    
        acc= (tp + tn)/(tp + fp + fn + tn)
        nuevos_datos = [epochs, acc]  # Reemplaza esto con tus propios datos
        apendear_csv(archivo_csv, nuevos_datos)
        #print("true positives")
        #print(tp)
        #print("false positives")
        #print(fp)
        #print("true negatives")
        #print(tn)
        #print("false negatives")
        #print(fn)
        #print("acc")
        #acc= (tp + tn)/(tp + fp + fn + tn)
        #print(acc)
        
# Función para abrir y leer un archivo CSV
def abrir_csv(archivo):
    try:
        with open(archivo, mode='r') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        return data
    except FileNotFoundError:
        print(f"El archivo {archivo} no existe.")
        return []

# Función para agregar datos al archivo CSV
def apendear_csv(archivo, nuevos_datos):
    try:
        with open(archivo, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(nuevos_datos)
        #print(f"Se agregaron los datos {nuevos_datos} al archivo {archivo}.")
    except FileNotFoundError:
        print(f"El archivo {archivo} no existe.")
    except Exception as e:
        print(f"Ocurrió un error al agregar datos al archivo: {str(e)}")
        
def graficar_datos():
    data = abrir_csv("datos3.csv")
    
    if not data:
        return
    
    # Separa los datos en ejes x e y
    ejes_x = [float(row[0]) for row in data[1:]]  # Ignora la primera fila (encabezados)
    ejes_y = [float(row[1]) for row in data[1:]]  # Ignora la primera fila (encabezados)
    
    # Crea el gráfico sin marcadores
    plt.figure(figsize=(8, 6))
    plt.plot(ejes_x, ejes_y, linestyle='-')  # 'marker' establecido en 'none'
    plt.title('Gráfico de Datos')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.grid(True)
    
    # Muestra el gráfico
    plt.show()
    
def graficar_datos_sin_marcadores():
    data = abrir_csv("datos3.csv")
    
    if not data:
        return
    
    # Separa los datos en ejes x e y
    ejes_x = [float(row[0]) for row in data[1:]]  # Ignora la primera fila (encabezados)
    ejes_y = [float(row[1]) for row in data[1:]]  # Ignora la primera fila (encabezados)
    
    # Crea el gráfico sin marcadores
    plt.figure(figsize=(8, 6))
    plt.plot(ejes_x, ejes_y, linestyle='-')  # 'marker' establecido en 'none'
    plt.title('Gráfico de Datos')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.grid(True)
    
    # Muestra el gráfico
    plt.show()
    
if __name__ == "__main__":
    main()