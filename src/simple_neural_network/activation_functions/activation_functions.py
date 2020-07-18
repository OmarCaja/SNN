import numpy as np

from simple_neural_network.constants import constants


class ActivationFunctions:

    @staticmethod
    def step_function(x):
        return np.heaviside(x, constants.ACTIVATION_FUNCTIONS.get('STEP_FUNCTION_VALUE'))

    @staticmethod
    def sigmoid_function(x):
        return 1 / (1 + np.exp(-x))
