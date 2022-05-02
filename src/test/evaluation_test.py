import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
from evaluation import Evaluator


class TestEvaluator(unittest.TestCase):
    def test_perfect_score(self):
        # to test for realistic runtime too; 
        # if the data set was balanced, this would be a realistic size of data set and dist of classes
        pred, gold = [], []
        for i in range(650):
            pred += [i for _ in range(77)]
            gold += [i for _ in range(77)]
        evaluator = Evaluator(gold, pred)
        self.assertEqual(evaluator.accuracy(), 1)
        self.assertEqual(evaluator.micro_fscore(), 1)
        self.assertEqual(evaluator.macro_fscore(), 1)

    def test_zero_score(self):
        pred = [i for i in range(650)]
        # gold set shifted by one
        gold = [i+1 for i in range(650)]
        gold[len(gold)-1] = 0
        evaluator = Evaluator(gold, pred)
        self.assertEqual(evaluator.accuracy(),0)
        self.assertEqual(evaluator.micro_fscore(), 0)
        self.assertEqual(evaluator.macro_fscore(), 0)

    def test_functionality(self):
        evaluator = Evaluator([1,2,0],[1,2,1])
        self.assertAlmostEqual(evaluator.accuracy(), 2/3)
        self.assertEqual(evaluator.precision_per_class(), {0: 0, 1: 0.5, 2: 1.0})
        self.assertEqual(evaluator.recall_per_class(), {0: 0, 1: 1.0, 2: 1.0})
        self.assertAlmostEqual(evaluator.fscore_per_class(), {0: 0, 1: 2/3, 2: 1.0})
        self.assertEqual(evaluator.macro_precision(), 0.5)
        self.assertAlmostEqual(evaluator.micro_precision(), 2/3)
        self.assertAlmostEqual(evaluator.macro_recall(), 2/3)
        self.assertAlmostEqual(evaluator.micro_recall(), 2/3)
        self.assertAlmostEqual(evaluator.macro_fscore(), 0.57, delta=0.002)
        self.assertAlmostEqual(evaluator.micro_fscore(), 2/3)

if __name__ == "__main__":
    unittest.main()