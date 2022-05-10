from collections import defaultdict, Counter
import itertools
import math
from typing import List

class TfIdf():
    """Converts a collection of documents into a tf-idf feature
    matrix representation

    tf is weighted with term frequency (classic).
    idf is weighted using inverse document frequency (classic).
    """

    def fit(self, docs: List[List[str]]):
        """Learns vocabulary and idf values from input documents

        Args:
            docs (List[List[str]]): list of documents represented as list of str
        """
        # Counts the number of documents that contain term t
        document_freq_vocab = defaultdict(int)
        for doc in docs:
            for term in set(doc):
                document_freq_vocab[term] += 1


        # Create vocabulary of terms with corresponding indexes
        self._vocab = {term:index for index, term in 
                       enumerate(document_freq_vocab.keys())}

        # Idf computed with inverse document frequency
        idf = defaultdict(int)
        for term, freq in document_freq_vocab.items():
            idf[term] = -math.log(freq / len(docs))

        self._idf = idf


    def transform(self, docs: List[List[str]]) -> List[List[float]]:
        """Transform input documents into tf-idf document-term matrix

        Args:
            docs (List[List[str]]): list of documents represented as list of str

        Returns:
            List[List[float]]: tf-idf document-term matrix
        """
        tfidf_matrix = []
        
        for doc in docs:
            vector = [0] * len(self._vocab)

            for term, freq in Counter(doc).items():
                # ignore term if not in vocabulary
                if term not in self._vocab:
                   continue 

                term_index = self._vocab[term]
                tf_value = freq / len(doc)
                idf_value = self._idf.get(term)
                vector[term_index] = tf_value * idf_value

            tfidf_matrix.append(vector)

        return tfidf_matrix
    
    def fit_transform(self, docs: List[List[str]]) -> List[List[float]]:
        """Learns vocabulary and idf values from input documents and 
        returns tf-idf document-term matrix

        TODO: This joins the fit() and transform() methods, not optimized.

        Args:
            docs (List[List[str]]): list of documents represented as list of str

        Returns:
            List[List[float]]: tf-idf document-term matrix
        """
        self.fit(docs)
        return self.transform(docs)