import numpy as np

from simple_neural_network.activation_functions.activation_functions_enum import ActivationFunctionsEnum
from simple_neural_network.constants import constants


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

    def __calculate_output(self, input_values):
        input_values = np.append(1, input_values)
        if self.activation_function is ActivationFunctionsEnum.STEP_FUNCTION:
            return np.heaviside(self.__calculate_propagation(input_values),
                                constants.ACTIVATION_FUNCTIONS.get('STEP_FUNCTION_VALUE'))
        elif self.activation_function is ActivationFunctionsEnum.IDENTITY_FUNCTION:
            return self.__calculate_propagation(input_values)
