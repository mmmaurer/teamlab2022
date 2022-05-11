from typing import Any, List
from __future__ import annotations


class Vector():
    """General class for feature representation.

    Allows concatinating multiple data representations into one vector
    representation. This will allow us to test out different feature
    embeddings easily.
    """

    def __init__(self, inputs: List[Any]):
        """Takes in inputs and concatinates them into one
        vector representation.

        Args:
            inputs (List[Any]): 
        """
        pass

    def cosine_similarity(self, vector: Vector) -> float:
        """Calculates the cosine similarity between the class and
        input vector. Value 1 means vectors are the same.

        Args:
            vector (Vector): input vector 

        Returns:
            float: value in range [0, 1]
        """
        pass

    def euclidean_distance(self, vector: Vector) -> float:
        """Calculates the euclidean_distance between the class and
        input vector. Value 0 means the vector are in the same position.

        Args:
            vector (Vector): input vector 

        Returns:
            float: value in range [0, inf]
        """
        pass