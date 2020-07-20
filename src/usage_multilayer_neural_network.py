from simple_neural_network.activation_functions.activation_functions_enum import ActivationFunctionsEnum
from simple_neural_network.loss_functions.loss_functions_enum import LossFunctionsEnum
from simple_neural_network.neural_systems.multilayer_neural_network.multilayer_neural_network import \
    MultilayerNeuralNetwork
from simple_neural_network.utilities.data_loader.csv_data_loader import CSVDataLoader
from simple_neural_network.utilities.neural_system_picker.neural_system_picker import NeuralSystemPicker
from simple_neural_network.utilities.normalization.normalization import Normalization

train_samples = CSVDataLoader.load_samples('./data/mnist/mnist_train_40K_samples.csv', ';', False)
train_labels = CSVDataLoader.load_labels('./data/mnist/mnist_train_40K_labels.csv', ';', False)
test_samples = CSVDataLoader.load_samples('./data/mnist/mnist_test_10K_samples.csv', ';', False)
test_labels = CSVDataLoader.load_labels('./data/mnist/mnist_test_10K_labels.csv', ';', False)

train_samples_normalized = Normalization.z_score(train_samples)
test_samples_normalized = Normalization.z_score(test_samples)

multilayer_neural_network = MultilayerNeuralNetwork([[784],
                                                     [20, ActivationFunctionsEnum.SIGMOID_FUNCTION],
                                                     [10, ActivationFunctionsEnum.SIGMOID_FUNCTION]])
multilayer_neural_network.train(train_samples_normalized, train_labels, LossFunctionsEnum.MSE_FUNCTION, 0.1, 30)

# NeuralSystemPicker.save_neural_system('./serialized_objects/multilayer_neural_network', multilayer_neural_network)
# multilayer_neural_network = NeuralSystemPicker.load_neural_system('./serialized_objects/multilayer_neural_network.snn')

print(multilayer_neural_network.calculate_error_rate(test_samples_normalized, test_labels))
