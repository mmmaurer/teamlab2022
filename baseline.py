import re
import datasets
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),
                             'src'))
from classifiers.knn import Knn
from evaluation.evaluation import Evaluator
from preprocessing.preprocessing import Preprocessor
from data_representations.bow import BOW

def tokenize(lyrics, keep_punc=False):
        """Tokenizes the raw data into a list of words in the lyrics by first
        cleaning the special newline characters and then splitting the string
        on the whitespaces. Depending on the value of self.keep_punc it removes
        or keeps punctuation.
        Args:
            lyrics (string): Lyrics of a song to be tokenized
        Returns:
            list(string): list of words in the lyrics
        """
        cleaned = lyrics.replace(" NEWLINE ", ' ').replace(" NEWLINE\n", '')
        if not keep_punc:  # remove punctuation
            return re.sub(r'[^\w\s]', '', cleaned).split(' ')
        else:
            return cleaned.split(' ')

def load_preprocess(n):
    """Loads and preprocesses the data:
    - Loading data from disk
    - Tokenizing
    - Converting to BOW
    - Filtering for classes

    Args:
        n (int): Number of classes to keep in the subset.
    Returns:
        train pd.Dataframe, test pd.Dataframe, artists list(string): Training and test datasets, list of artists
    """
    data_folder = "./data/"

    # Read and load training data file
    train_file = data_folder + "songs_train.txt"

    train = []
    with open(train_file, 'r') as f:
        for line in f.readlines():
            # The three values for each line are joined together with the tab '\t' character
            artist, title, lyrics = line.split('\t')

            train.append([artist, title, lyrics])
    
    # Read and load testing data file
    test_file = data_folder + "songs_test.txt"
    
    test = [] 
    with open(test_file, 'r') as f:
        for line in f.readlines():
            # The three values for each line are joined together with the tab '\t' character
            artist, title, lyrics = line.split('\t')

            test.append([artist, title, lyrics])

    # Creating datasets
    data_train = pd.DataFrame(train, columns=['artist', 'title', 'lyrics'])
    data_test = pd.DataFrame(test, columns=['artist', 'title', 'lyrics'])
    train_ds = datasets.Dataset.from_pandas(data_train)

    # Tokenize and create BOWs
    data_train["tokenized"] = [tokenize(lyrics) for lyrics in data_train.lyrics]
    data_train["bow"] = data_train.tokenized.apply(BOW)
    data_test["tokenized"] = [tokenize(lyrics) for lyrics in data_test.lyrics]
    data_test["bow"] = data_test.tokenized.apply(BOW)

    # Filter for only the n artists to experiment on
    artists = list(dict.fromkeys(train_ds["artist"]))[:n]
    train = data_train[data_train.artist.isin(artists)]
    test = data_test[data_test.artist.isin(artists)]

    return train, test, artists

if __name__ == "__main__":
    # Reading command line arguments
    n = int(sys.argv[1])
    processes = int(sys.argv[2])
    
    # Loading and preprocessing datasets
    train, test, artists = load_preprocess(n)
    
    # Convert artist names to indices
    label_to_num = {artist:i for i, artist in enumerate(artists)}

    # Convert training and test dataset BOWs and labels into the format
    # the classifier needs
    training_examples = list(train.bow)
    training_labels = [label_to_num[label] for label in list(train.artist)]
    test_examples = list(test.bow)
    test_labels = [label_to_num[label] for label in list(test.artist)]
    
    classifier = Knn(input=training_examples, targets=training_labels, multi_process=processes)
    
    for i in range(25):  # Going through different numbers of neighbors
        curr_k = i+1
        predictions = classifier.predict(test_examples, k=curr_k, measure="jaccard")
        
        evaluator = Evaluator(test_labels, predictions)
        
        curr_acc = evaluator.accuracy()

        # Logging the results
        with open(f"./baseline_results_{n}_classes.csv", "a+") as f:
            f.write(f"knn-bow;{n};{len(train)};{len(test)};{curr_acc};{curr_k}\n")
