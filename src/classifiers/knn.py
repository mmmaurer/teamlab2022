import itertools
from typing import List
import sys
import os
import concurrent.futures

sys.path.append(os.path.join(os.path.dirname(__file__), '..',
                             'data_representations'))
from vector import Vector


class Knn():
    """The K-nearest neighbor classifier for artist classification.
    Using vector representations.

    This class is implemented to allow for multiprocess execution.
    """

    def __init__(self,
                 input: List[Vector],
                 targets: List[int],
                 multi_process=1) -> None:
        """ TODO: Input typing will change to a general representation.

        Args:
            input (typing.List[BOW]): training examples used by the model.
            targets (typing.List[int]): labels corresponding to the training
                                        examples.
            multi_process (int, optional): number of processes to use when
                                           predicting. If > 1, the algorithm
                                           switches to multiprocessing.
                                           Defaults to 1.

        Raises:
            ValueError: raised when targets length not equal to input list
                        length.
        """
        if len(input) != len(targets):
            error = f"""Input ({len(input)}) and targets {len(targets)}
                    not same dimensions."""
            raise ValueError(error)

        # self.data = multiprocessing.Manager().list(input)
        # self.targets = multiprocessing.Manager().list(targets)

        self.multi_process = multi_process
        self.data = input
        self.targets = targets

    def _predict(self, input: List[Vector], k, measure) -> List[int]:
        predictions = []
        for x in input:
            distances = []
            # Calculate distance between x and all examples in training set
            for i, example in enumerate(self.data):
                distance = example.distance(x, measure=measure)
                distances.append([i, distance])

            # Sorts distances and picks k closest examples
            k_picks = sorted(distances, key=lambda ex: ex[1])[:k]

            # Using saved indexes get corresponding labels
            labels = [self.targets[example[0]] for example in k_picks]

            label = max(labels, key=labels.count)
            predictions.append(label)
      
        return predictions

    def _predict_multiprocess(self,
                              input: List[Vector],
                              k,
                              measure) -> List[int]:
        # Function used for splitting input into chunks for processes
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
                
                
        chunk_size = int(len(input) / self.multi_process)
        with concurrent.futures.ProcessPoolExecutor(self.multi_process) as ex:
            futures = []
            predictions = []

            for examples in chunks(input, chunk_size):
                futures.append(ex.submit(self._predict, examples, k, measure))

            concurrent.futures.wait(futures)
            predictions += list(itertools.chain(*[future.result() for future in futures]))

        return predictions

    def predict(self, input: List[Vector], k=5, measure="cosine") -> List[int]:
        """Predict classification for a list of input examples.

        If the value of variable 'multi_process' > 1, the algorithm will make
        predictions using multiple processes.

        Args:
            input (typing.List[BOW]): input examples we want to predict.
            k (int, optional): number of nearest neighbours to compare.
                               Defaults to 5.
            measure (string, optional): Distance measure to use.
                                        Defaults to "cosine".

        Raises:
            TypeError: raised when input is not of same class type as
                       examples in the model.

        Returns:
            typing.List[int]: list of predictions.
        """

        if not isinstance(input[0], Vector) == isinstance(self.data[0],
                                                          Vector):
            raise TypeError("Input and model data types are not of same class")

        if self.multi_process > 1:
            predictions = self._predict_multiprocess(input, k, measure)
        else:
            predictions = self._predict(input, k, measure)

        return predictions
