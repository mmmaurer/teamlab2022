import sys
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '../..',
                             'data_representations'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from classifiers.knn import Knn
from evaluation.evaluation import Evaluator
from vector import Vector
from data_representations import BOW


class TestKnn(unittest.TestCase):
    def test_perfect_bow_prediction(self):
        """Test perfect prediction of the classifier using BOW by trying to
        predict the training data.

        Since it's predicting itself, it should return a 100% evaluation score.
        """
        training = [
            BOW(["Chickens", "be", "like", "that", "sometimes"]),
            BOW(["Moms", "be", "like", "that", "sometimes"]),
            BOW(["Dads", "be", "like", "that", "sometimes"]),
            BOW(["Dads", "be", "like", "that", "sometimes"]),
            BOW(["Grandmas", "be", "like", "that", "sometimes"]),
        ]
        labels = [1, 2, 3, 3, 4]

        classifier = Knn(training, labels)
        predictions = classifier.predict(training, k=1, measure='jaccard')

        evaluator = Evaluator(labels, predictions)
        self.assertEqual(evaluator.accuracy(), 1)

    def test_bow_majority_misclassification(self):
        """Test misclassification of Knn by setting k high enough that the incorrect
        majority class is selected for prediction.
        """

        # Because we have duplicate of "Dads be like that sometimes",
        # setting k=3 will result in the duplicates will become the
        # majority class
        training = [
            BOW(["Chickens", "be", "like", "that", "sometimes"]),
            BOW(["Moms", "be", "like", "that", "sometimes"]),
            BOW(["Dads", "be", "like", "that", "sometimes"]),
            BOW(["Dads", "be", "like", "that", "sometimes"]),
            BOW(["Grandmas", "be", "like", "that", "never"]),
        ]
        labels = [1, 2, 3, 3, 4]

        testing_inputs = [
            BOW(['Dads', 'like', 'nothing']),
            BOW(['Grandmas', 'like', 'nothing']),
            # This example will result in majority misclassification
            BOW(['Grandmas', 'Dads', 'like', 'never'])
        ]
        testing_labels = [3, 4, 4]

        classifier = Knn(training, labels)
        predictions = classifier.predict(testing_inputs, k=3, measure='jaccard')

        evaluator = Evaluator(testing_labels, predictions)
        self.assertAlmostEqual(evaluator.accuracy(), 2/3)


if __name__ == "__main__":
    unittest.main()
