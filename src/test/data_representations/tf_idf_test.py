import unittest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data_representations.tf_idf import TfIdf


class TestTfIdf(unittest.TestCase):
    def setUp(self) -> None:
        training_docs = [
            ['something', 'new', 'for', 'today'],
            ['something', 'new', 'maybe', 'today'],
        ]

        self.tfidf = TfIdf()
        self.tfidf.fit(training_docs)

    def test_tfidf_dictionary_has_indexes(self):
        """Tests if the tf-idf object created a vocabulary that holds the indexes of
        terms
        """
        vocab_indexes_gold = list(range(5))
        vocab_indexes = list(self.tfidf._vocab.values())

        self.assertCountEqual(vocab_indexes_gold, vocab_indexes)

    def test_tfidf_terms_in_dictionary(self):
        """Tests if the tf-idf object created a correct vocabulary of all
        the terms from the training data
        """
        vocab_terms_gold = ['for', 'maybe', 'new', 'something', 'today']
        vocab_terms = list(self.tfidf._vocab.keys())

        self.assertCountEqual(vocab_terms_gold, vocab_terms)

    def test_tfidf_fit_transform(self):
        """Tests the function tfidf.fit_transform on learning a new vocab and correctly
        transforming input documents into term-weight matrix.
        """

        training_docs = [
            ['I', 'do', 'want', 'ice', 'cream'],
            ['I', 'dont', 'want', 'ice', 'cream'],
        ]

        matrix = self.tfidf.fit_transform(training_docs)

        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(matrix[0]), 6)

        # Checks if unique term in one document has a weight
        term_index = self.tfidf._vocab['do']
        tfidf_value = matrix[0][term_index]
        self.assertGreater(tfidf_value, 0)


if __name__ == "__main__":
    unittest.main()
