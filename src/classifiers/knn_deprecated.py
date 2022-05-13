from typing import List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..',
                             'data_representations'))
from data_representations import BOW


class Knn():
    """The K-nearest neighbor classifier for artist classification.
    Using set BOW.

    The algorithm makes the classification
    """

    def __init__(self, input: List[BOW], targets: List[int]) -> None:
        """ TODO: Input typing will change to a general representation.

        Args:
            input (typing.List[BOW]): training examples used by the model.
            targets (typing.List[int]): labels corresponding to the training
                                        examples.

        Raises:
            ValueError: raised when targets length not equal to input list
                        length.
        """
        if len(input) != len(targets):
            error = f"""Input ({len(input)}) and targets {len(targets)}
                    not same dimensions."""
            raise ValueError(error)

        self.data = input
        self.targets = targets

    def predict(self, input: List[BOW], k=5, measure="tversky",
                alpha=1, beta=1) -> List[int]:
        """Predict classification for a list of input examples.

        Args:
            input (typing.List[BOW]): input examples we want to predict.
            k (int, optional): number of nearest neighbours to compare.
                               Defaults to 5.
            measure (string, optional): Distance measure to use.
                                        Defaults to "tversky".
            alpha (float, optional): Alpha value for Tversky index.
                                     Defaults to 1.
            beta (float, optional): Beta value for Tversky index.
                                    Defaults to 1.

        Raises:
            TypeError: raised when input is not of same class type as
                       examples in the model.

        Returns:
            typing.List[int]: list of predictions.
        """

        if not type(input[0]) is type(self.data[0]):
            raise TypeError("Input and model data types are not of same class")

        predictions = []
        for x in input:
            distances = []
            # Calculate distance between x and all examples in training set
            for i, example in enumerate(self.data):
                distance = example.distance(x, measure=measure,
                                            alpha=alpha, beta=beta)
                distances.append([i, distance])

            # Sorts distances and picks k closest examples
            k_picks = sorted(distances, key=lambda ex: ex[1])[:k]

            # Using saved indexes get corresponding labels
            labels = [self.targets[example[0]] for example in k_picks]

            label = max(labels, key=labels.count)
            predictions.append(label)

        return predictions
