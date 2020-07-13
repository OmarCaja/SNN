import numpy as np

from simple_neural_network.activation_functions.activation_functions_enum import ActivationFunctionsEnum
from simple_neural_network.constants import constants
from simple_neural_network.neuron.neuron import Neuron


class BinaryClassifier:

    def __init__(self, number_of_inputs):
        self.__neuron = Neuron(number_of_inputs, ActivationFunctionsEnum.STEP_FUNCTION)
        self.__learning_rate = constants.BINARY_CLASSIFIER.get('LEARNING_RATE_DEFAULT_VALUE')
        self.__max_epochs = constants.BINARY_CLASSIFIER.get('MAX_EPOCHS_DEFAULT_VALUE')
        self.__miss_classified_samples_per_epoch = []

    @property
    def learning_rate(self):
        return self.__learning_rate

    @property
    def max_epochs(self):
        return self.__max_epochs

    @property
    def miss_classified_samples_per_epoch(self):
        return self.__miss_classified_samples_per_epoch

    @property
    def iterations(self):
        return len(self.__miss_classified_samples_per_epoch)

    @property
    def weights(self):
        return self.__neuron.weights

    def train(self, samples, labels, learning_rate, max_epochs):
        self.__learning_rate = learning_rate
        self.__max_epochs = max_epochs
        iteration = 0

        while True:
            well_classified_samples = 0
            miss_classified_samples = 0

            for sample, label in zip(samples, labels):
                error = label - self.__neuron.calculate_output(sample)
                if error != 0:
                    self.__neuron.weights += (self.learning_rate * error * np.append(1, sample))
                    miss_classified_samples += 1
                else:
                    well_classified_samples += 1

            iteration += 1
            self.miss_classified_samples_per_epoch.append(miss_classified_samples)

            if iteration == self.max_epochs or well_classified_samples == samples.shape[0]:
                break

    def classify(self, sample):
        return self.__neuron.calculate_output(sample)

    def calculate_error(self, samples, labels):
        errors = 0

        for sample, label in zip(samples, labels):
            if self.classify(sample) != label[0]:
                errors += 1

        return errors / len(labels)
