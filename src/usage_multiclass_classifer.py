from simple_neural_network.neural_systems.multiclass_classifier.multiclass_classifier import MulticlassClassifier
from simple_neural_network.utilities.data_loader.csv_data_loader import CSVDataLoader

train_samples = CSVDataLoader.load_samples('./data/mnist/mnist_train_40K_samples.csv', ';', False)
train_labels = CSVDataLoader.load_labels('./data/mnist/mnist_train_40K_labels.csv', ';', False)
test_samples = CSVDataLoader.load_samples('./data/mnist/mnist_test_10K_samples.csv', ';', False)
test_labels = CSVDataLoader.load_labels('./data/mnist/mnist_test_10K_labels.csv', ';', False)

multiclass_classifier = MulticlassClassifier(784, 10)
multiclass_classifier.train(train_samples, train_labels, 0.5, 20)

print(multiclass_classifier.calculate_error(test_samples, test_labels))
