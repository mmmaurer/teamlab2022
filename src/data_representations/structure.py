from collections import defaultdict, Counter
import math
from typing import List


class Structure():
    """Gives structural information about the processed documents
    """
    def __init__(self, docs: List[List[str]]):
        """

        Args:
            docs (List[List[str]]): list of documents that will be processed
        """

        self.docs = docs

    @property
    def number_lines(self) -> List[int]:
        """The number of lines each document holds

        Returns:
            List[int]: line size for each document
        """
        return [len(doc) for doc in self.docs]

    @property
    def doc_length(self) -> List[int]:
        """The length of each document

        Returns:
            List[int]: length of each document
        """
        doc_lengths = []
        for doc in self.docs:
            doc_l = sum(len(line) for line in doc)
            doc_lengths.append(doc_l)
        
        return doc_lengths