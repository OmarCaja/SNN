from simple_neural_network.neuron.neuron import Neuron


class MultilayerNeuralNetwork:
    dimension = 0
    activation_function = 1

    def __init__(self, layers_definition):
        self.__number_of_classes = layers_definition[-1][MultilayerNeuralNetwork.dimension]
        self.__layers = [[Neuron(layers_definition[layer - 1][MultilayerNeuralNetwork.dimension],
                                 layers_definition[layer][MultilayerNeuralNetwork.activation_function])
                          for _ in range(layers_definition[layer][MultilayerNeuralNetwork.dimension])]
                         for layer in range(1, len(layers_definition))]
