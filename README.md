# Simple Neural Network

Simple neural network es una librería escrita en python con fines didácticos.

En esta se encuentran implementados 3 tipos de sistemas neuronales como son:
* Clasificador binario.
* Clasificador multicalse.
* Redes neuronales multicapa.

Este notebook tiene como objetivo presentar la forma en la que están programados cada uno de los elementos de esta librería.

Se hará referencia a la ubicación de cada uno de los elementos dentro de la estructura de la librería de la siguiente forma: `simple_neural_network/package(s)/file.py`.
Siendo `simple_neural_network` la raíz de la librería.


## Dependencias

Definiremos en primer lugar las dependencias necesarias para poder utilizar esta liibrería y haremos distinción entre las que ya vienen por defecto con python y las que hay que descargar de forma explícita.

* Paquetes de python:
    * csv: dedicado a la lectura de datos desde ficheros csv.
    * pickle: utilizado para guardar y cargas los sistemas neuronales en disco y poder reutilizarlos.
    * enum: utilizado para la definición de enums, como por ejemplo los tipos de funciones de adtivación. De esta forma es más cómodo seleccionar una función u otra sin cometer errores, otra librerías utilizan strings.


* Paquetes externos:
    * matplotlib: utiliada para reprsentar información de forma gráfica.
    * numpy: esencial para trabajar con matrices y poder realiar operaciones con estas de una forma más cómda y eficiente.


```python
import csv
import pickle
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
```

## Constantes

Ubicación: `simple_neural_network/constants/constants.py`

Se crea este archivo para un manejo más sencillo y localizado de los valores constantes de los elementos que componen la librería.



```python
# Utilities
OBJECTS_SAVED_EXTENSION = '.snn'

# Activation functions
STEP_FUNCTION_VALUE = 0

# Binary classifier
RATE_DEFAULT_VALUE = 1
MAX_ITERATIONS_DEFAULT_VALUE = 100
```

## Lectura de datos desde archivos CSV

Ubicación: `simple_neural_network/utilities/data_loader/csv_data.py`

Una de las utilidades de la librería es un módulo para la lectura de datos desde archivos `.csv`, ya que estos achivos son muy comunes para la representación de sets de datos.

Se puede observar como se dispone de dos métodos públicos estáticos capaces de generar un array de numpy tanto para datos como para etiquetas. Siendo el tipo de datos de `np.float32` y `np.intc` respectivamente.
Este array que devuelve cada método se generea a partir de los datos del archivo csv que se pasa como parámetro.

Las dimensiones del array de datos será NxM, siendo N el número de filas y M el número de columnas del archivo `.csv`.
Y Nx1 para el array de etiquetas.

Nótese como se elimina la primera línea del archivo `.csv`, ya que contiene la descripción de las columnas.

Ejemplo:

`samples.csv`
```
# sepal_length, # petal_length
5.1, 1.4
4.9, 1.4
4.7, 1.3
4.6, 1.5
```
`CSVData.load_samples(samples.csv)` -> [[5.1, 1.4], [4.9, 1.4], [4.7, 1.3], [4.6, 1.5]]

`labels.csv`
```
# labels (0 = Iris-setosa, 1 = Iris-versicolor)
0
0
0
0
```
`CSVData.load_labels(labels.csv)` -> [[0], [0], [0], [0]]


```python
class CSVData:

    @staticmethod
    def __load_data(path_to_csv_file):
        with open(path_to_csv_file) as samples_csv_file:
            data = csv.reader(samples_csv_file)

            data_list = []
            for row in data:
                data_list.append(row)
            data_list.pop(0)

            return data_list

    @staticmethod
    def load_samples(path_to_csv_file):
        return np.array(CSVData.__load_data(path_to_csv_file), dtype=np.float32)

    @staticmethod
    def load_labels(path_to_csv_file):
        return np.array(CSVData.__load_data(path_to_csv_file), dtype=np.intc)
```

## Guardado y carga de sistemas neuronales

Ubicación: `simple_neural_network/utilities/neural_systems_picker/neural_systems_picker.py`

Otra utilidad implementada es la de guardado y carga de sistemas neuronales. Para ello se definen dos métodos estáticos para el guardado y la carga de sistema neuronales en disco.
Se puede observar como la extensión de los archivos queda definida en la constante `OBJECTS_SAVED_EXTENSION = '.snn'`.

Esto nos permite reutilizar redes ya entrenenadas para hacer un uso más eficiente de las mismas.


```python
class NeuralSystemsPicker:

    @staticmethod
    def save_neural_system(file_name, neural_system):
        file = open(file_name + OBJECTS_SAVED_EXTENSION, 'wb')
        pickle.dump(neural_system, file)

    @staticmethod
    def load_neural_system(file_name):
        file = open(file_name, 'rb')
        return pickle.load(file)
```

## Enums

Ubicación: `simple_neural_network/activation_functions/activation_functions_enum.py`

En este archivo se definen los diferentes tipos de funciones de activación, una forma más cómoda de trabajar en lugar de utilizar strings o ints para su identificación.

Su uso se realiza de la sguiente forma: `ActivationFunctionsEnum.STEP_FUNCTION`


```python
class ActivationFunctionsEnum(Enum):
    STEP_FUNCTION = 1
```

## Neurona

Ubicación: `simple_neural_network/neuron/neuron.py`

La clase `Neuron` es el elemento principal de esta librería, ya que formará parte de todos los sistemas neuronales que se definirán a continuación.

Elementos que componen la clase neurona:

* Constructor:
    * `Neuron(number_of_inputs, activation_function)`: recibe como argumentos el número de entradas, y el tipo de función de activación.
    Al número de entradas se le suma 1 que hace referencia a la constante que multiplica al bias de los pesos.


* Aributos:
    * `__weights` (pesos): son los pesos de la neurona.
    * `__activation_function` (función de activación): el tipo de función de activación definida para la neurona. Hace referencia al enum `ActivationFunctionsEnum` definido anteriomente.


* Funciones:
    * `weights`: getter y setter del atributo `weights`.
    * `activation_function`: getter del atributo `activation_function`. En este caso no se define un setter ya que la definición de este atributo se realiza en el constructor y no debe cambiar durante la ejecución.
    * `__calculate_propagation`: función de propagación de los valores de entrada y los pesos de la neurona, en este caso se trata del producto escalar entre ambos vectores.
    Es un método privado ya que solo debe hacer uso de este la función de activación.
    * `calculate_output`: función de propagación de la neurona, encargada de calcular la salida de la misma en función del tipo de función de activación definida.


```python
class Neuron:

    def __init__(self, number_of_inputs, activation_function):
        self.__weights = np.random.random(number_of_inputs + 1)
        self.__activation_function = activation_function

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__weights = value

    @property
    def activation_function(self):
        return self.__activation_function

    def __calculate_propagation(self, input_values):
        return np.dot(self.weights, input_values)

    def calculate_output(self, input_values):
        input_values = np.append(1, input_values)
        if self.activation_function is ActivationFunctionsEnum.STEP_FUNCTION:
            return np.heaviside(self.__calculate_propagation(input_values), STEP_FUNCTION_VALUE)
```

## Calsificador binario

Ubicación: `simple_neural_network/neural_systems/binary_classifier/binary_classifier.py`

Es el primer sistema neuronal definido por ser el más básico y que menos elementos requiere. Este sistema es capaz de definir una frontera lineal entre dos clases.

Elementos que componen al clasificador binario:

* Constructor:
    * `BinaryClassifier(number_of_inputs)`: recibe como argumento un único entero que indica el número de entradas de la neurona (dimensión de las muestras), sin contar la constante = 1 que multiplica al valor del bias.

* Atributos:
    * `__neuron`: atributo de tipo `Neuron`definido anteriormente, es el encargado de realizar la clasificación y el objeto de entrenamiento.
    * `__rate`: atributo que define la velocidad de aprendizaje del algoritmo, por defecto su valor es 1, `RATE_DEFAULT_VALUE = 1`.
    * `__max_iterations`: atributo que define la cantidad máxima de iteraciones que realizará el algoritmo en caso de que las muestras no sean linealmente separables, por defecto su valor es 100 `MAX_ITERATIONS_DEFAULT_VALUE = 100`.
    * `__miss_classified_samples_per_iteration`: es una lista donde cada elemento corresponde al número de muestras mal clasificadas en la iteración pos + 1 de la lista.
    Es decir `__miss_classified_samples_per_iteration[x]` corresponde al número de muestras mal clasificadas en la iteración x + 1.


* Funciones:
    * `rate`: getter del atributo `rate`. En este caso no se define un setter ya que la definición de este atributo se realiza en el método `train(rate, max_iterations)` y no debe cambiar durante la ejecución.
    * `max_iterations`: getter del atributo `max_iterations`. En este caso no se define un setter ya que la definición de este atributo se realiza en el método `train(rate, max_iterations)` y no debe cambiar durante la ejecución.
    * `miss_classified_samples_per_iteration`: getter del atributo `miss_classified_samples_per_iteration`. En este caso no se define un setter ya que este atributo se genera en tiempo de ejecución.
    * `weights`: getter del atributo `__neuron.weights` definido en la clase `Neuron`.
    * `train(samples, labels, rate, max_iterations)`: función encargada de realizar el entrenamiento del sistema, el algoritmo utilizado es el del perceptrón con la modalidad del atributo rate para la velocidad de entrenamiento.

    Recibe como argumentos:

        * `samples`: un `np.array` de tipo `np.float32`para las muestras.
        * `labels`: un `np.array` de tipo `np.intc` para las etiquetas.
        Las muestras tendrán una dimensión NxM y las etiquetas Nx1 donde N hace referencia al número de muestras,
        y donde la etiqueta label[x] es la correspondiente a la muestra sample[x].

        * `rate`: velocidad de aprendizaje del algoritmo.
        * `max_iterations`: número máximo de iteraciones del algoritmo.

    Se puede observar como el algoritmo se ejecuta hasta que o bien haya clasificado correctamente todas las muestras en una misma iteración o se alcance el máximo número de iteraciones.

    * `classify(sample)`: función encargada de clasificar una muestra, de vuelve 0 o 1.


```python
class BinaryClassifier:

    def __init__(self, number_of_inputs):
        self.__neuron = Neuron(number_of_inputs, ActivationFunctionsEnum.STEP_FUNCTION)
        self.__rate = RATE_DEFAULT_VALUE
        self.__max_iterations = MAX_ITERATIONS_DEFAULT_VALUE
        self.__miss_classified_samples_per_iteration = []

    @property
    def rate(self):
        return self.__rate

    @property
    def max_iterations(self):
        return self.__max_iterations

    @property
    def miss_classified_samples_per_iteration(self):
        return self.__miss_classified_samples_per_iteration

    @property
    def iterations(self):
        return len(self.__miss_classified_samples_per_iteration)

    @property
    def weights(self):
        return self.__neuron.weights

    def train(self, samples, labels, rate, max_iterations):
        self.__rate = rate
        self.__max_iterations = max_iterations
        iteration = 0

        while True:
            well_classified_samples = 0
            miss_classified_samples = 0

            for sample, label in zip(samples, labels):
                error = label - self.__neuron.calculate_output(sample)
                if error != 0:
                    self.__neuron.weights += (self.__rate * error * np.append(1, sample))
                    miss_classified_samples += 1
                else:
                    well_classified_samples += 1

            iteration += 1
            self.miss_classified_samples_per_iteration.append(miss_classified_samples)

            if iteration == self.max_iterations or well_classified_samples == samples.shape[0]:
                break

    def classify(self, sample):
        return self.__neuron.calculate_output(sample)
```

## Ejemplo de uso del clasificador binario

Ubicación: `usage_binary_classifier.py`

En este archivo se realiza un ejemplo de uso del clasificador binario con un set de datos real como es el de iris-setosa e iris-versicolor.
Se dispone de dos archivos .csv en el directorio `/data`, `iris_virginica_samples.csv` e `iris_virginica_labels.csv`, estos archivos contienen la longitud de pétalos y sépalos de cada tipo de flor y su clase.

En primer lugar se cargan los datos `samples` y `labels` haciendo uso de `CSVData`.

Instanciamos un `BinaryClassifier(samples.shape[1])`, donde `samples.shape[1] = dimensión de la primera muestra = 2` y lo entrenamos con `binary_classifier.train(samples, labels, 0.8, 20)` con un `rate = 0.8` y `max_iterations = 20`.

Tras entrenar el sistema y a modo de ejemplo, lo guardamos en disco `NeuralSystemsPicker.save_neural_system('binary_classifier', binary_classifier)`
y lo volvemos a cargar `binary_classifier = NeuralSystemsPicker.load_neural_system('binary_classifier.snn')`.

Por último representamos los datos obtenidos:
* En primer lugar imprimimos las muestras y la intersección del hiperplano generado por el clasificador con el plano XY para Z = 0.
* Por último imprimimos el número de errores por iteración obtenidos durante el entrenamiento.


```python
samples = CSVData.load_samples('./data/iris_virginica_train_60_samples.csv')
labels = CSVData.load_labels('./data/iris_virginica_train_60_labels.csv')

binary_classifier = BinaryClassifier(samples.shape[1])
binary_classifier.train(samples, labels, 0.8, 20)

NeuralSystemsPicker.save_neural_system('binary_classifier', binary_classifier)
binary_classifier = NeuralSystemsPicker.load_neural_system('binary_classifier.snn')

plt.scatter(np.array(samples[:50, 0]), np.array(samples[:50, 1]), marker='o', label='Setosa')
plt.scatter(np.array(samples[50:, 0]), np.array(samples[50:, 1]), marker='x', label='Versicolor')
plt.xlabel('Petal length')
plt.ylabel('Sepal length')
plt.legend()

weights = binary_classifier.weights
x = np.linspace(4, 7.5, 100)
y = (-1) * (weights[1] * x + weights[0]) / weights[2]
plt.plot(x, y, '-r', linewidth=2)

axes = plt.gca()
axes.set_xlim(4, 7.5)
axes.set_ylim(0.5, 5.5)
plt.show()

plt.plot(binary_classifier.miss_classified_samples_per_iteration)
plt.axis([0, 6, 0, 5])
plt.ylabel('Miss classified samples')
plt.xlabel('Iteration')
plt.show()
```


![png](output_16_0.png)



![png](output_16_1.png)

