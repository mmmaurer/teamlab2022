from typing import List

class TfIdf():
    """Converts a collection of documents into a tf-idf feature matrix representation
    """

    def __init__(self, documents: List[List[str]]) -> None:
        self.fit(documents)


    def fit(self, documents: List[List[str]]):
        """Learns idf values from input documents

        Args:
            documents (List[List[str]]): list of documents represented as list of str
        """
        if len(documents) == 0:
            return
        pass

    def fit_transform(self, documents: List[List[str]]) -> List[List[float]]:
        """Learns idf values from input documents and returns document-term matrix

        This joins the fit() and transform() methods

        Args:
            documents (List[List[str]]): list of documents represented as list of str

        Returns:
            List[List[float]]: tf-idf document-term matrix
        """
        self.fit(documents)
        return self.transform(documents)

    def transform(self, documents: List[List[str]]) -> List[List[float]]:
        """Transform input documents into document-term matrix using learned idf values

        Args:
            documents (List[List[str]]): list of documents represented as list of str

        Returns:
            List[List[float]]: tf-idf document-term matrix
        """
        pass

    