import itertools
from typing import List, Union
import sys
import os
import concurrent.futures

sys.path.append(os.path.join(os.path.dirname(__file__), '..',
                             'data_representations'))
from vector import Vector
from bow import BOW


class Knn():
    """The K-nearest neighbor classifier for artist classification
    using vector representations.

    This class is implemented to allow for multiprocess execution.
    """

    def __init__(self,
                 input: List[Union[Vector, BOW]],
                 targets: List[int],
                 multi_process=1) -> None:
        """

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

        self.multi_process = multi_process
        self.data = input
        self.targets = targets

    def _predict(self,
                 input: List[Union[Vector, BOW]],
                 k,
                 measure,
                 alpha,
                 beta) -> List[int]:
        """Internal method that maked a prediction of the input.

        Args:
            input (List[Union[Vector, BOW]]): input examples we want to
                                              predict.
            k (int): number of nearest neighbours to compare.
            measure (string): Distance measure to use.
            alpha (float): Alpha value for Tversky index.
            beta (float): Beta value for Tversky index.

        Returns:
            List[int]: list of predictions.
        """
        predictions = []
        for x in input:
            distances = []
            # Calculate distance between x and all examples in training set
            for i, example in enumerate(self.data):
                if measure == 'tversky':
                    distance = example.distance(x,
                                                measure=measure,
                                                alpha=alpha,
                                                beta=beta)
                else:
                    distance = example.distance(x, measure=measure)
                distances.append([i, distance])

            # Sorts distances and picks k closest examples
            k_picks = sorted(distances, key=lambda ex: ex[1])[:k]

            # Using saved indexes get corresponding labels
            labels = [self.targets[example[0]] for example in k_picks]

            label = max(labels, key=labels.count)
            predictions.append(label)

        return predictions

    def _multiprocess_predict(self,
                              input: List[Union[Vector, BOW]],
                              k,
                              measure,
                              alpha,
                              beta) -> List[int]:
        """Internal method that maked a prediction of the input using
        multiple processes.

        Actually just calls self._predict().

        This method is very memory intensive, since every process
        has it's own memory copy of the main program. So use wisely.

        Args:
            input (List[Union[Vector, BOW]]): input examples we want to
                                              predict.
            k (int): number of nearest neighbours to compare.
            measure (string): Distance measure to use.
            alpha (float): Alpha value for Tversky index.
            beta (float): Beta value for Tversky index.

        Returns:
            List[int]: list of predictions.
        """

        # Function used for splitting input into chunks for processes
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        # Decides how many examples each process will have, so that
        # it's evenly distributed amoung n processes
        chunk_size = int(len(input) / self.multi_process)

        with concurrent.futures.ProcessPoolExecutor(self.multi_process) as ex:
            futures = []
            predictions = []

            for examples in chunks(input, chunk_size):
                # The ex.submit method starts up a new process and returns
                # a Future object refering to newly created process.
                #
                # ex.submit also saves the order of predictions, so there's
                # no need to keep extra track of orders
                futures.append(ex.submit(self._predict,
                                         examples,
                                         k,
                                         measure,
                                         alpha,
                                         beta))

            # waits for all processes to finish predictions to be able to
            # join them together into a final list
            concurrent.futures.wait(futures)

            predictions += list(itertools.chain(*[future.result()
                                                  for future in futures]))

        return predictions

    def predict(self,
                input: List[Union[Vector, BOW]],
                k=5,
                measure="cosine",
                alpha=1.0,
                beta=1.0) -> List[int]:
        """Predict classification for a list of input examples.

        If the value of variable 'multi_process' > 1, the algorithm will make
        predictions using multiple processes for performance gains.

        Args:
            input (List[Union[Vector, BOW]]): input examples we want to
                                              predict.
            k (int, optional): number of nearest neighbours to compare.
                               Defaults to 5.
            measure (string, optional): Distance measure to use.
                                        Defaults to "cosine".
            alpha (float, optional): Alpha value for Tversky index.
                                     Defaults to 1.
            beta (float, optional): Beta value for Tversky index.
                                    Defaults to 1.

        Raises:
            TypeError: raised when input is not of same class type as
                       examples in the model.

        Returns:
            List[int]: list of predictions.
        """

        if not isinstance(input[0], Vector) == isinstance(self.data[0],
                                                          Vector):
            raise TypeError("Input and model data types are not of same class")

        if self.multi_process > 1:
            predictions = self._multiprocess_predict(input,
                                                     k,
                                                     measure,
                                                     alpha,
                                                     beta)
        else:
            predictions = self._predict(input,
                                        k,
                                        measure,
                                        alpha,
                                        beta)

        return predictions
