import typing


class BOW():
    """Bag of word class represented by sets
    """
    def __init__(self, text: typing.List[typing.AnyStr]):
        self.rep = set(text)

    def similarity(self, other, measure="tversky", alpha=1, beta=1):
        """Similarity between two BOWs.
        Choose between:
            - Naive approach
            - Jaccard
            - Overlap
            - Sørensen-Dice
            - Tversky

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the similarity score within [0,1],
                   the higher, the more similar
        """
        measures = {
            "tversky": self.__tversky,
            "dsc": self.__dsc,
            "jaccard":  self.__jaccard,
            "overlap": self.__overlap,
            "naive": self.__naive
        }
        if measure == "tversky":
            return self.__tversky(other, alpha=alpha, beta=beta)
        else:
            return measures[measure](other)

    def distance(self, other, measure="tversky", alpha=1, beta=1):
        """The distance between two BOWs.

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the distance score within [0,1], the lower, the nearer
        """
        return 1 - self.similarity(other, measure=measure,
                                   alpha=alpha, beta=beta)

    def __naive(self, other):
        """We define a naive similarity as n^(-1) with n being the number of words
        in the intersection of two bags of words

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the distance score within [0,1], the lower, the nearer
        """
        n = len(self.rep.intersection(other.rep))
        # if no element in the intersection, just use maximum distance
        if n == 0:
            return 1
        else:
            return (1/n)

    def __jaccard(self, other):
        """Jaccard index/similarity coefficient for two BOWs
        J(A,B) = |(A intersection B)| / |(A union B)|

        Special case of Tversky index with alpha=beta=1

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the Jaccard index
        """
        return self.__tversky(other, alpha=1, beta=1)

    def __dsc(self, other):
        """Sørensen-Dice coefficient for two BOWs
        DSC(A,B) = 2 |(A intersection B)| / (|A| + |B|)

        Special case of Tversky index with alpha=beta=0.5

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the DSC index
        """
        return self.__tversky(other, alpha=0.5, beta=0.5)

    def __tversky(self, other, alpha=1, beta=1):
        """Tversky index for two BOWs
        S(A,B) = |(A intersection B)| /
                (|(A intersection B)| + alpha*(|A-B|) +  beta*(|B-A|)

        alpha=beta=1 -> Jaccard index
        alpha=beta=0.5 -> Sørensen-Dice coefficient

        Tversky measures with alpha+beta=1 are of
        special interest (according to wiki)

        Args:
            other (BOW): the other bag of words to compare against
            alpha (int, optional): Defaults to 1.
            beta (int, optional): Defaults to 1.

        Returns:
            float: the Tversky index
        """
        intersection = self.rep.intersection(other.rep)

        n = (len(intersection) +
             abs(alpha*len(self.rep - other.rep)) +
             abs(beta*len(other.rep - self.rep)))
        if n == 0:  # both BOWs are empty
            return 0  # minimum Tversky index
        else:
            return len(intersection)/n

    def __overlap(self, other):
        """Overlap coefficient for two BOWs
        overlap(A, B) = |(A intersection B)| / min(|A|,|B|)

        Args:
            other (BOW): the other bag of words to compare against

        Returns:
            float: the overlap coefficient
        """
        n = min(len(self.rep), len(other.rep))
        if n == 0:  # both BOWs are empty
            return 0  # minimum overlap coefficient
        else:
            return len(self.rep.intersection(other.rep))/n

    def __repr__(self):
        return str(self.rep)
