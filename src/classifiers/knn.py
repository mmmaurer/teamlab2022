import typing

from data_representations.data_representations import BOW

class Knn():
    """Implementation of the K-nearest neighbor classifier for artist classification.
    """

    def __init__(self, input: typing.List[BOW], targets: typing.List[int]) -> None:
        """ TODO: Input typing will change to a general representation.

        Args:
            input (typing.List[BOW]): training examples used by the model
            targets (typing.List[int]): labels corresponding to the training examples. 
        """
        
        self.data = input
        self.targets = targets


    def predict(self, input: typing.List[BOW], k=5) -> typing.List[int]:
        """Predict classification for a list of input examples.

        Args:
            input (typing.List[BOW]): input examples we want to predict.
            k (int, optional): number of nearest neighbours to compare. Defaults to 5.

        Returns:
            typing.List[int]: list of predictions
        """
        return [1]