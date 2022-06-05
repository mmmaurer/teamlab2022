import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../',
                             'data_representations'))
from bow import BOW


class testBOW(unittest.TestCase):
    def setUp(self):
        """Setting up testcases
        """
        self.bow1 = BOW(["I", "am", "very", "pleased"])
        self.bow2 = BOW(["I", "am", "quite", "unhappy"])

    def test_jaccard(self):
        # J(A,B) = |(A intersection B)| / |(A union B)| = 2/6 = 1/3
        self.assertAlmostEqual(self.bow1.similarity(self.bow2,
                                                    measure="jaccard"), 1/3)
        # distance = 1 - similarity
        self.assertAlmostEqual(self.bow1.distance(self.bow2,
                                                  measure="jaccard"), 2/3)

    def test_dsc(self):
        # DSC(A,B) = 2 |(A intersection B)| / (|A| + |B|) = 4/8 = 1/2
        self.assertEqual(self.bow1.similarity(self.bow2, measure="dsc"), 1/2)
        # distance = 1 - similarity
        self.assertEqual(self.bow1.distance(self.bow2, measure="dsc"), 1/2)

    def test_tversky(self):
        # With alpha = beta = 0.1
        # S(A,B) = |(A intersection B)| /
        #        (|(A intersection B)| + alpha*(|A-B|) +  beta*(|B-A|)
        #        = 2 / (2 + 0.2 + 0.2) = 2/2.4 = 5/6
        self.assertAlmostEqual(self.bow1.similarity(self.bow2,
                                                    measure="tversky",
                                                    alpha=0.1,
                                                    beta=0.1), 5/6)
        # distance = 1 - similarity
        self.assertAlmostEqual(self.bow1.distance(self.bow2,
                                                  measure="tversky",
                                                  alpha=0.1,
                                                  beta=0.1), 1/6)

    def test_overlap(self):
        # overlap(A, B) = |(A intersection B)| / min(|A|,|B|) = 2/4 = 1/2
        self.assertEqual(self.bow1.similarity(self.bow2,
                                              measure="overlap"), 1/2)
        # distance = 1 - similarity
        self.assertEqual(self.bow1.distance(self.bow2,
                                            measure="overlap"), 1/2)

    def test_edgecases(self):
        bow0 = BOW([])  # empty bow
        # Asserting for empty second bow similarity is minimal
        self.assertEqual(self.bow1.similarity(bow0, "jaccard"), 0)
        self.assertEqual(self.bow1.similarity(bow0, "dsc"), 0)
        self.assertEqual(self.bow1.similarity(bow0, "overlap"), 0)
        self.assertEqual(self.bow1.similarity(bow0, "tversky"), 0)
        # and distance is maximal
        self.assertEqual(self.bow1.distance(bow0, "jaccard"), 1)
        self.assertEqual(self.bow1.distance(bow0, "dsc"), 1)
        self.assertEqual(self.bow1.distance(bow0, "overlap"), 1)
        self.assertEqual(self.bow1.distance(bow0, "tversky"), 1)


if __name__ == '__main__':
    unittest.main()
