import sys
import os
from classifiers.knn import Knn
from evaluation.evaluation import Evaluator
from preprocessing.preprocessing import Preprocessor


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    # Read dataset
    filepath_train = "./data/songs_train.txt"
    dataset_train = Preprocessor(filepath=filepath_train, read_limit=10000)

    filepath_test = "./data/songs_test.txt"
    dataset_test = Preprocessor(filepath=filepath_test, read_limit=100)

    # Create numerical representations of labels for mapping
    label_to_num = {artist:i for i, artist in enumerate(set(dataset_train.artists) | set(dataset_test.artists))}
    num_to_label = {value:key for key, value in label_to_num.items()}

    # Initiate Knn classifier
    training_examples = dataset_train.BOW()
    training_labels = [label_to_num[label] for label in dataset_train.artists]
    classifier = Knn(training_examples, training_labels)

    # Test predictions
    test_examples = dataset_test.BOW()
    test_labels = [label_to_num[label] for label in dataset_test.artists]
    predictions = classifier.predict(test_examples, k=4)

    # Run evaluation of algorithms performance
    evaluator = Evaluator(test_labels, predictions)
    print("Accuracy:",evaluator.accuracy())