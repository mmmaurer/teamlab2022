from __future__ import annotations
from typing import Any, List
import math


class Vector():
    """General class for feature representation.

    Allows concatinating multiple data representations into one vector
    representation. This will allow us to test out different feature
    embeddings easily.
    """

    def __init__(self, inputs: List[List[float]]):
        """Takes in inputs and concatinates them into one
        vector representation.

        Args:
            inputs (List[List[float]]): list of vectors
        """
        self._vector = []
        for input in inputs:
            self._vector += input

    @property
    def vector(self):
        return self._vector

    @property
    def magnitute(self):
        return math.sqrt(sum([math.pow(i, 2) for i in self._vector]))

    def __iter__(self):
        for val in self._vector:
            yield val

    def distance(self, other, measure="cosine"):
        """Distance between class and input vector.
        Choose between cosine and euclidean.

        Args:
            other (Vector): input vector to compare against.
            measure (str, optional): Measure for comparison.
                                     Defaults to "cosine".

        Raises:
            NotImplementedError: raises when not implemented 
                                 measure gets chosen.

        Returns:
            float: distance/similarity measure
        """
        if measure == "cosine":
            return self.__cosine_similarity(other)
        elif measure == "euclidean":
            return self.__euclidean_distance(other)
        else:
            error = f"{measure} not implemented (yet)."
            raise NotImplementedError(error)

    def __cosine_similarity(self, other: Vector) -> float:
        """Calculates the cosine similarity between the class and
        input vector. Value 1 means vectors are the same.

        Args:
            other (Vector): input vector to compare against.

        Returns:
            float: value in range [0, 1]
        """
        dot_product = sum([i * j for i, j in zip(self, other)])
        return dot_product / (self.magnitute * other.magnitute)

    def __euclidean_distance(self, other: Vector) -> float:
        """Calculates the euclidean_distance between the class and
        input vector. Value 0 means the vector are in the same position.

        Args:
            other (Vector): input vector to compare against.

        Returns:
            float: value in range [0, inf]
        """
        _sum = sum([math.pow(i - j, 2) for i, j in zip(self, other)])
        return math.sqrt(_sum)
