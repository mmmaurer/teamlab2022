import typing

from data_representations.data_representations import BOW

class Knn():
    """The K-nearest neighbor classifier for artist classification.

    The algorithm makes the classification
    """

    def __init__(self, input: typing.List[BOW], targets: typing.List[int]) -> None:
        """ TODO: Input typing will change to a general representation.

        Args:
            input (typing.List[BOW]): training examples used by the model.
            targets (typing.List[int]): labels corresponding to the training examples.

        Raises:
            ValueError: raised when targets length not equal to input list length.
        """
        if len(input) != len(targets):
            raise ValueError(f"Input and targets lists are not the same dimension. input len={len(input)}, targets len={len(targets)}")
        
        self.data = input
        self.targets = targets



    def predict(self, input: typing.List[BOW], k=5) -> typing.List[int]:
        """Predict classification for a list of input examples.

        Args:
            input (typing.List[BOW]): input examples we want to predict.
            k (int, optional): number of nearest neighbours to compare. Defaults to 5.

        Raises:
            TypeError: raised when input is not of same class type as examples in the model.

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
                distance = example.distance(x)
                distances.append([i, distance])

            # Sorts distances and picks k closest examples
            k_picks = sorted(distances, key=lambda ex:ex[1])[:k]

            # Using saved indexes get corresponding labels
            labels = [self.targets[example[0]] for example in k_picks]

            label = max(labels, key=labels.count)
            predictions.append(label)

        
        return predictions