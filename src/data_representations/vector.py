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

    def cosine_similarity(self, vector: Vector) -> float:
        """Calculates the cosine similarity between the class and
        input vector. Value 1 means vectors are the same.

        Args:
            vector (Vector): input vector 

        Returns:
            float: value in range [0, 1]
        """
        dot_product = sum([i * j for i, j in zip(self, vector)])
        return dot_product / (self.magnitute * vector.magnitute)

    def euclidean_distance(self, vector: Vector) -> float:
        """Calculates the euclidean_distance between the class and
        input vector. Value 0 means the vector are in the same position.

        Args:
            vector (Vector): input vector 

        Returns:
            float: value in range [0, inf]
        """
        _sum = sum([math.pow(i - j, 2) for i, j in zip(self, vector)])
        return math.sqrt(_sum)
