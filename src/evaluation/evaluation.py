import typing


class Evaluator():
    """Acts as an evaluator object for a given pair of predictions and their
    corresponding ground truth
    """
    def __init__(self, gold: typing.List[int], pred: typing.List[int]):
        """
        Args:
            gold (typing.List[int]): a list of integers with the index of the
                                     true class of an example
            pred (typing.List[int]): a list of integers with the index of the
                                     predicted class of an example
        """
        self.gold = gold
        self.pred = pred
        self.classes = set(gold)
        # we need to know the number of classes for micro and macro metrics
        self.n_classes = len(self.classes)
        # Precalculating the following since we might want to output more than
        # one metric so might aswell
        tp, fn, fp, tn = self.instances_per_class()
        self.tp_per_class, self.fn_per_class = tp, fn
        self.fp_per_class, self.tn_per_class = fp, tn

    def instances_per_class(self):
        """Calculates TP, FP, TN, FN per class

        Returns:
            tp, fp, tn, fn: dicts with respective number of (mis-)
                            classifications per class
        """
        tp, fp, tn, fn = {}, {}, {}, {}
        for i in self.classes:
            tp[i] = 0
            fp[i] = 0
            tn[i] = 0
            fn[i] = 0

        # for each pair of gold label and prediction
        for g, p in zip(self.gold, self.pred):
            # for each class
            for c in self.classes:
                # gold label and prediction match
                # and match the class -> TP
                if g == p and p == c:
                    tp[c] += 1
                # gold label and prediction match
                # and don't match the class -> TN
                elif g == p and p != c:
                    tn[c] += 1
                # gold label and prediction don't match
                # but gold matches class -> FN
                elif g == c and c != p:
                    fn[c] += 1
                # gold label and prediction don't match
                # but prediction and class match -> FP
                elif g != p and p == c:
                    fp[c] += 1
                # Nothing matches -> TN
                else:
                    tn[c] += 1

        return tp, fn, fp, tn

    def accuracy(self):
        """Accuracy = TP_total / (TP_total + FN_total + FP_total + TN_total)
                    = TP_total / size_dataset

        Returns:
            float: accuracy
        """
        return len([1 for i, j in zip(self.gold, self.pred) if i == j]) \
            / len(self.gold)
        # the length of gold and pred should be equal, so either
        # would be fine here

    def precision_per_class(self):
        """Precisions per class: TP / (TP + FP)

        Returns:
            dict(int:float): precision per class in the gold data
        """
        # if-condition to prevent division by zero in cases where there
        # is no instance classified as the class or both TP and FN are zero
        prec_dict = {}
        for i in self.classes:
            if (i not in self.pred or (self.tp_per_class[i] ==
                                       self.fn_per_class[i] == 0)):
                prec_dict[i] = 0
            else:
                prec_dict[i] = self.tp_per_class[i] / \
                    (self.tp_per_class[i]+self.fp_per_class[i])
        return prec_dict

    def recall_per_class(self):
        """Recalls per class: TP / (TP + FN)

        Returns:
            dict(int:float): recall per class in the gold data
        """
        # if-condition to prevent division by zero in cases where there is
        # no instance classified as the class or both TP and FN are zero
        rec_dict = {}
        for i in self.classes:
            if (i not in self.pred or (self.tp_per_class[i] ==
                                       self.fn_per_class[i] == 0)):
                rec_dict[i] = 0
            else:
                rec_dict[i] = (self.tp_per_class[i] /
                               (self.tp_per_class[i]+self.fn_per_class[i]))
        return rec_dict

    def fscore_per_class(self):
        """F-Score F1 = (2PR)/(P+R) per class

        Returns:
            dict(int:float): F-Score F1 per glass
        """
        # if-condition to prevent division by zero in cases where there
        # is no instance classified as the class
        f_dict = {}
        for i in self.classes:
            if (i not in self.pred or
                (self.precision_per_class()[i] ==
                 self.recall_per_class()[i] == 0)):
                f_dict[i] = 0
            else:
                f_dict[i] = ((2 * self.precision_per_class()[i] *
                              self.recall_per_class()[i])
                             /
                             (self.precision_per_class()[i] +
                              self.recall_per_class()[i]))
        return f_dict

    def macro_precision(self):
        """For n classes:
        Macro averaged precision
            = (precision_1 + ... + precision_n) / num_classes

        Returns:
            float: macro averaged precision
        """
        return sum(self.precision_per_class().values())/self.n_classes

    def macro_recall(self):
        """For n classes:
        Macro averaged recall = (recall_1 + ... + recall_n) / num_classes

        Returns:
            float: macro averaged recall
        """
        return sum(self.recall_per_class().values())/self.n_classes

    def macro_fscore(self):
        """F1 = (2PR)/(P+R)

        Returns:
            float: macro averaged F-Score
        """
        # precalculating denominator to prevent division by zero if there's
        # no correctly predicted instance and thus P and R are zero
        denominator = (self.macro_precision() + self.macro_recall())
        if denominator == 0:
            return 0
        else:
            return ((2 * self.macro_precision() * self.macro_recall()) /
                    denominator)

    def micro_precision(self):
        """For n classes:
        Micro averaged precision = (TP_1 + ... + TP_n)
                                    /
                                   ( (TP_1 + ... + TP_n) +
                                   (FP_1 + ... + FP_n) )

        Returns:
            float: micro averaged precision
        """
        return (sum(self.tp_per_class.values()) /
                (sum(self.tp_per_class.values()) +
                sum(self.fp_per_class.values())))

    def micro_recall(self):
        """For n classes:
        Micro averaged precision = (TP_1 + ... + TP_n) /
                                   ( (TP_1 + ... + TP_n) +
                                     (FN_1 + ... + FN_n) )

        Returns:
            float: micro averaged recall
        """
        return (sum(self.tp_per_class.values()) /
                (sum(self.tp_per_class.values()) +
                sum(self.fn_per_class.values())))

    def micro_fscore(self):
        """F1 = (2PR)/(P+R)

        Returns:
            float: micro averages F-Score
        """
        # precalculating denominator to prevent division by zero if there's
        # no correctly predicted instance and thus P and R are zero
        denominator = (self.micro_precision() + self.micro_recall())
        if denominator == 0:
            return 0
        else:
            return ((2 * self.micro_precision() * self.micro_recall()) /
                    denominator)


if __name__ == '__main__':
    evaluator = Evaluator([1, 2, 0, 0], [1, 2, 1, 1])
    print(evaluator.tp_per_class)
    print(evaluator.fn_per_class)
    print(evaluator.fp_per_class)
    print(evaluator.tn_per_class)
    print(evaluator.accuracy())
    print(evaluator.precision_per_class())
    print(evaluator.recall_per_class())
    print(evaluator.fscore_per_class())
    print(evaluator.macro_precision())
    print(evaluator.micro_precision())
    print(evaluator.macro_recall())
    print(evaluator.micro_recall())
    print(evaluator.macro_fscore())
    print(evaluator.micro_fscore())
