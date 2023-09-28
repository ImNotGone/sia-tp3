import csv
import matplotlib.pyplot as plt

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

# Nombre del archivo CSV con los datos
archivo_csv = "datos4.csv"

# Abre el archivo CSV
data = abrir_csv(archivo_csv)

# Divide los datos en dos conjuntos
Train = [(float(row[0]), float(row[1])) for i, row in enumerate(data) if i % 2 == 0]
noise = [(float(row[0]), float(row[1])) for i, row in enumerate(data) if i % 2 != 0]

# Extrae los ejes x e y de cada conjunto
x1, y1 = zip(*Train)
x2, y2 = zip(*noise)

# Crea el gráfico con colores diferentes
plt.figure(figsize=(8, 6))
plt.plot(x1, y1, linestyle='-', label='Normal', color='blue')
plt.plot(x2, y2, linestyle='-', label='Noise', color='orange')
plt.title('Gráfico de Datos')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.grid(True)
plt.legend()  # Muestra la leyenda

# Muestra el gráfico
plt.show()