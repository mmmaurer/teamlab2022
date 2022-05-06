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

    def jaccard(self, other):
        """Jaccard index/similarity coefficient for two BOWs
        J(A,B) = |(A intersection B)| / |(A union B)|

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the Jaccard index
        """
        n = len(self.rep.union(other.rep))
        if n == 0: # both BOWs are empty
            return 0 # minimum Jaccard index
        else:
            return len(self.rep.intersection(other.rep))/n

    def dsc(self, other):
        """Sørensen-Dice coefficient for two BOWs
        DSC(A,B) = 2 |(A intersection B)| / (|A| + |B|)

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the DSC index
        """
        n = (len(self.rep) + len(other.rep))
        if n == 0: # both BOWs are empty
            return 0 # minimum Sørensen-Dice coefficient
        else:
            return 2 * len(self.rep.intersection(other.rep))/n

    def tversky(self, other, alpha=1, beta=1):
        """Tversky index for two BOWs
        S(A,B) = |(A intersection B)| / (|(A intersection B)| + alpha*(|A-B|) +  beta*(|B-A|)

        alpha=beta=1 -> Jaccard index
        alpha=beta=0.5 -> Sørensen-Dice coefficient
        Tversky measures with alpha+beta=1 are of special interest (according to wiki)

        Args:
            other (BOW): the other bag of words to compare against
            alpha (int, optional): Defaults to 1.
            beta (int, optional): Defaults to 1.

        Returns:
            float: the Tversky index
        """
        n = (len(self.rep.intersection(other.rep)) + abs(alpha*len(self.rep - other.rep)) + abs(beta*len(other.rep - self.rep)))
        if n == 0: # both BOWs are empty
            return 0 # minimum Tversky index
        else:
            return len(self.rep.intersection(other.rep))/n

    def overlap(self, other):
        """Overlap coefficient for two BOWs
        overlap(A, B) = |(A intersection B)| / min(|A|,|B|)

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the overlap coefficient
        """
        n = min(len(self.rep), len(other.rep))
        if n == 0: # both BOWs are empty
            return 0 # minimum overlap coefficient
        else:
            return len(self.rep.intersection(other.rep))/n

    def __repr__(self):
        return str(self.rep)

