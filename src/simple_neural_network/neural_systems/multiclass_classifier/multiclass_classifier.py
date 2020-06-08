import numpy as np

from simple_neural_network.activation_functions.activation_functions_enum import ActivationFunctionsEnum
from simple_neural_network.constants import constants
from simple_neural_network.neuron.neuron import Neuron


class MulticlassClassifier:

    def __init__(self, number_of_inputs, number_of_classes):
        self.__number_of_classes = number_of_classes
        self.__neurons = [Neuron(number_of_inputs, ActivationFunctionsEnum.IDENTITY_FUNCTION)
                          for _ in range(number_of_classes)]
        self.__rate = constants.RATE_DEFAULT_VALUE
        self.__max_iterations = constants.MAX_ITERATIONS_DEFAULT_VALUE
        self.__miss_classified_samples_per_iteration = []

    @property
    def number_of_classes(self):
        return self.__number_of_classes

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
        return [neuron.weights for neuron in self.__neurons]

    def train(self, samples, labels, rate, max_iterations):
        self.__rate = rate
        self.__max_iterations = max_iterations
        iteration = 0

        while True:
            well_classified_samples = 0
            miss_classified_samples = 0

            for sample, label in zip(samples, labels):

                well_classifier_value = self.__neurons[label[0]].calculate_output(sample)
                error = False

                for classifier in [classifier for classifier in range(self.number_of_classes)
                                   if classifier != label[0]]:
                    wrong_classifier_value = self.__neurons[classifier].calculate_output(sample)

                    if wrong_classifier_value > well_classifier_value:
                        self.__neurons[classifier].weights -= (self.rate * np.append(1, sample))
                        error = True

                    if error:
                        self.__neurons[label[0]].weights += (self.rate * np.append(1, sample))

                if error:
                    miss_classified_samples += 1
                else:
                    well_classified_samples += 1

            iteration += 1
            self.miss_classified_samples_per_iteration.append(miss_classified_samples)

            if iteration == self.max_iterations or well_classified_samples == samples.shape[0]:
                break

    def classify(self, sample):
        results = [neuron.calculate_output(sample) for neuron in self.__neurons]
        return results.index(max(results))

    def calculate_error(self, samples, labels):
        errors = 0

        for sample, label in zip(samples, labels):
            if self.classify(sample) != label[0]:
                errors += 1

        return errors / len(labels)
