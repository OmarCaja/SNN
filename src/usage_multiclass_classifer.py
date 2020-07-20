from simple_neural_network.neural_systems.multiclass_classifier.multiclass_classifier import MulticlassClassifier
from simple_neural_network.utilities.data_loader.csv_data_loader import CSVDataLoader
from simple_neural_network.utilities.neural_system_picker.neural_system_picker import NeuralSystemPicker

train_samples = CSVDataLoader.load_samples('./data/mnist/mnist_train_40K_samples.csv', ';', False)
train_labels = CSVDataLoader.load_labels('./data/mnist/mnist_train_40K_labels.csv', ';', False)
test_samples = CSVDataLoader.load_samples('./data/mnist/mnist_test_10K_samples.csv', ';', False)
test_labels = CSVDataLoader.load_labels('./data/mnist/mnist_test_10K_labels.csv', ';', False)

multiclass_classifier = MulticlassClassifier(784, 10)
multiclass_classifier.train(train_samples, train_labels, 0.5, 20)

NeuralSystemPicker.save_neural_system('./serialized_objects/multiclass_classifier', multiclass_classifier)
multiclass_classifier = NeuralSystemPicker.load_neural_system('./serialized_objects/multiclass_classifier.snn')

print(multiclass_classifier.calculate_error_rate(test_samples, test_labels))
