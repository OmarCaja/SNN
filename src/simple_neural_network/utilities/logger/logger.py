from simple_neural_network.constants import constants


class Logger:

    @staticmethod
    def print_error_rate_message(epoch, misclassified_samples, samples, error_rate):
        print(constants.ERROR_RATE.format(epoch=epoch, misclassified_samples=misclassified_samples,
                                          samples=samples, error_rate=error_rate))
