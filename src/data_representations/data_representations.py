import typing

class BOW():
    """Bag of word class represented by sets
    """
    def __init__(self, text: typing.List[typing.AnyStr]):
        self.rep = set(text)

    def similarity(self, other):
        """We define similarity as 1 - n^(-1) with n being the number of words
        in the intersection of two bags of words

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the similarity score within [0,1], the higher, the more similar
        """
        n = len(self.rep.intersection(other.rep))
        if n == 0: # if no elements in the intersection, just use minimum similarity
            return 0
        else:
            return 1 - (1/n)

    def distance(self, other):
        """We define similarity as n^(-1) with n being the number of words
        in the intersection of two bags of words

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the distance score within [0,1], the lower, the nearer
        """
        n = len(self.rep.intersection(other.rep))
        if n==0: # if no element in the intersection, just use maximum distance
            return 1
        else: 
            return (1/n)
    
    def __repr__(self):
        return str(self.rep)

