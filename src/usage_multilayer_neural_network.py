from simple_neural_network.activation_functions.activation_functions_enum import ActivationFunctionsEnum
from simple_neural_network.neural_systems.multilayer_neural_network.multilayer_neural_network import \
    MultilayerNeuralNetwork

multilayer_neural_network = MultilayerNeuralNetwork([[3],
                                                     [4, ActivationFunctionsEnum.IDENTITY_FUNCTION],
                                                     [5, ActivationFunctionsEnum.IDENTITY_FUNCTION],
                                                     [5, ActivationFunctionsEnum.IDENTITY_FUNCTION]])
