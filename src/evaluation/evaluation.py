import typing

# NOTE: I think it would make the code easier and more readable, if you don't have this huge amount of
# properties of tp_per_class, etc... and instance_per_class, but rather have simple functions
# that separately calculate TP, FP etc. and then just call those in the specific function.

# The computation will be bigger, but the code will look nicer, which I believe is more important, since
# it will make the code easier to debug and understand

# NOTE: as for calculating TP, FP, etc. per class: two options, etiher filter before sending it to the functions
# or add an extra optional parameter, like "filtr", that will filter inside before doing the calculations

class Evaluator():
    """Acts as an evaluator object for a given pair of predictions and their corresponding ground truth
    """
    def __init__(self, gold: typing.List[int], pred: typing.List[int]):
        """
        Args:
            gold (typing.List[int]): a list of integers with the index of the true class of an example
            pred (typing.List[int]): a list of integers with the index of the predicted class of an example
        """
        self.gold = gold
        self.pred = pred
        self.n_classes = len(set(gold)) # we need to know the number of classes for micro and macro metrics
        # Precalculating the following since we might want to output more than one metric so might aswell
        self.tp_per_class, self.fp_per_class, self.tn_per_class, self.fn_per_class = self.instances_per_class()


    def instances_per_class(self):
        """Calculates TP, FP, TN, FN per class

        Returns:
            tn, fp, tn, fn: dicts with respective number of (mis-)classifications per class
        """
        tp, fp, tn, fn = {},{},{},{}
        for i in range(self.n_classes):
            # Initializing per class to prevent key errors and checking if key for class is in keys of dict already
            tp[i], fp[i], tn[i], fn[i] = 0, 0, 0, 0

            # TODO: Check if it's not mixing j (from pred) with i (from gold) in the way you check for TP, FP,...
            for j, k in zip(self.pred, self.gold):
                if j==i:
                    if j==k:
                        tp[i] += 1
                    else:
                        fp[i] += 1
                if k==i:
                    if j==k:
                        tn[i] += 1
                    else:
                        fn[i] += 1
        return tn, fp, tn, fn
        
    def accuracy(self):
        """Accuracy = TP_total / (TP_total + FN_total + FP_total + TN_total) 
                    = TP_total / size_dataset

        Returns:
            float: accuracy
        """
        return len([1 for i,j in zip(self.gold, self.pred) if i == j])/len(self.gold) # the length of gold and pred should be equal, so either would be fine here

    def precision_per_class(self):
        """Precisions per class: TP / (TP + FP)

        Returns:
            dict(int:float): precision per class in the gold data
        """
        # if-condition to prevent division by zero in cases where there is no instance classified as the class or both TP and FN are zero
        prec_dict = {}
        for i in range (self.n_classes):
            if not i in self.pred or (self.tp_per_class[i] == self.fn_per_class[i] == 0):
                prec_dict[i] = 0
            else:
                prec_dict[i] = self.tp_per_class[i]/(self.tp_per_class[i]+self.fp_per_class[i])
        return prec_dict

    def recall_per_class(self):
        """Recalls per class: TP / (TP + FN)

        Returns:
            dict(int:float): recall per class in the gold data
        """
        # if-condition to prevent division by zero in cases where there is no instance classified as the class or both TP and FN are zero
        rec_dict = {}
        for i in range (self.n_classes):
            if not i in self.pred or (self.tp_per_class[i] == self.fn_per_class[i] == 0):
                rec_dict[i] = 0
            else:
                rec_dict[i] = self.tp_per_class[i]/(self.tp_per_class[i]+self.fn_per_class[i])
        return rec_dict

    def fscore_per_class(self):
        """F-Score F1 = (2PR)/(P+R) per class

        Returns:
            dict(int:float): F-Score F1 per glass
        """
        # if-condition to prevent division by zero in cases where there is no instance classified as the class
        return {i:(2 * self.precision_per_class()[i] * self.recall_per_class()[i]) / (self.precision_per_class()[i] + self.recall_per_class()[i]) if i in self.pred else 0 for i in range(self.n_classes)}
    
    def macro_precision(self):
        """For n classes:
        Macro averaged precision = (precision_1 + ... + precision_n) / num_classes

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
        # precalculating denominator to prevent division by zero if there's no correctly predicted instance and thus P and R are zero
        denominator = (self.macro_precision() + self.macro_recall())
        if denominator == 0:
            return 0
        else:
            return (2 * self.macro_precision() * self.macro_recall())/denominator

    def micro_precision(self):
        """For n classes:
        Micro averaged precision = (TP_1 + ... + TP_n) / ((TP_1 + ... + TP_n) + (FP_1 + ... + FP_n))

        Returns:
            float: micro averaged precision
        """
        return sum(self.tp_per_class.values())/(sum(self.tp_per_class.values()) + sum(self.fp_per_class.values()))

    def micro_recall(self):
        """For n classes:
        Micro averaged precision = (TP_1 + ... + TP_n) / ((TP_1 + ... + TP_n) + (FN_1 + ... + FN_n))

        Returns:
            float: micro averaged recall
        """
        return sum(self.tp_per_class.values())/(sum(self.tp_per_class.values()) + sum(self.fn_per_class.values()))

    def micro_fscore(self):
        """F1 = (2PR)/(P+R)

        Returns:
            float: micro averages F-Score
        """
        # precalculating denominator to prevent division by zero if there's no correctly predicted instance and thus P and R are zero
        denominator = (self.micro_precision() + self.micro_recall())
        if denominator == 0:
            return 0
        else:
            return (2 * self.micro_precision() * self.micro_recall())/denominator

if __name__ == '__main__':
    evaluator = Evaluator([1,2,0,0],[1,2,1,1])
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

