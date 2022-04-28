import typing

class Evaluator():
    '''
    Acts as an evaluator object for a given pair of predictions and their corresponding ground truth
    '''
    def __init__(self, gold: typing.List[int], pred: typing.List[int]):
        '''
        Types might change, refactoring depending on whether we can use numpy or not

        gold: a list of integers with the index of the true class of an example
        pred: a list of integers with the index of the predicted class of an example
        '''
        self.gold = gold
        self.pred = pred
        self.n_classes = len(set(gold)) # we need to know the number of classes for micro and macro metrics
        # Precalculating the following since we might want to output more than one metric so might aswell
        self.tp_per_class = {i:len([1 for j,k in zip(self.pred, self.gold) if j==i and j==k]) for i in range(self.n_classes)}
        self.fp_per_class = {i:len([1 for j,k in zip(self.pred, self.gold) if j==i and j!=k]) for i in range(self.n_classes)}
        self.fn_per_class = {i:len([1 for j,k in zip(self.pred, self.gold) if k==i and j==k]) for i in range(self.n_classes)}
        self.tn_per_class = {i:len([1 for j,k in zip(self.pred, self.gold) if k==i and j!=k]) for i in range(self.n_classes)}
        
    def accuracy(self):
        '''
        Accuracy = TP_total / (TP_total + FN_total + FP_total + TN_total) = TP_total / size_dataset
        '''
        return len([1 for i,j in zip(self.gold, self.pred) if i == j])/len(self.gold) # the length of gold and pred should be equal, so either would be fine here

    def precision_per_class(self):
        '''
        Precisions per class: TP / (TP + FP)
        Returns: 
        dict: precision per class in the gold data
        '''
        # if-condition to prevent division by zero in cases where there is no instance classified as the class
        return {i:self.tp_per_class[i]/(self.tp_per_class[i]+self.fp_per_class[i]) if i in self.pred else 0 for i in range(self.n_classes)}

    def recall_per_class(self):
        '''
        Recalls per class: TP / (TP + FN)
        Returns: 
        dict: recall per class in the gold data
        '''
        # if-condition to prevent division by zero in cases where there is no instance classified as the class
        return {i:self.tp_per_class[i]/(self.tp_per_class[i]+self.fn_per_class[i]) if i in self.pred else 0 for i in range(self.n_classes)}

    def fscore_per_class(self):
        '''
        '''
        # if-condition to prevent division by zero in cases where there is no instance classified as the class
        return {i:(2 * self.precision_per_class()[i] * self.recall_per_class()[i]) / (self.precision_per_class()[i] + self.recall_per_class()[i]) if i in self.pred else 0 for i in range(self.n_classes)}
    
    def macro_precision(self):
        '''
        For n classes:
        Macro averaged precision = (precision_1 + ... + precision_n) / num_classes
        '''
        return sum(self.precision_per_class().values())/self.n_classes

    def macro_recall(self):
        '''
        For n classes:
        Macro averaged recall = (recall_1 + ... + recall_n) / num_classes
        '''
        return sum(self.recall_per_class().values())/self.n_classes

    def macro_fscore(self):
        '''
        F1 = (2PR)/(P+R)
        '''
        return (2 * self.macro_precision() * self.macro_recall())/(self.macro_precision() + self.macro_recall())

    def micro_precision(self):
        '''
        For n classes:
        Micro averaged precision = (TP_1 + ... + TP_n) / ((TP_1 + ... + TP_n) + (FP_1 + ... + FP_n))
        '''
        return sum(self.tp_per_class.values())/(sum(self.tp_per_class.values()) + sum(self.fp_per_class.values()))

    def micro_recall(self):
        '''
        For n classes:
        Micro averaged precision = (TP_1 + ... + TP_n) / ((TP_1 + ... + TP_n) + (FN_1 + ... + FN_n))
        '''
        return sum(self.tp_per_class.values())/(sum(self.tp_per_class.values()) + sum(self.fn_per_class.values()))

    def micro_fscore(self):
        '''
        F1 = (2PR)/(P+R)
        '''
        return (2 * self.micro_precision() * self.micro_recall())/(self.micro_precision() + self.micro_recall())

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

