import re
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_representations'))
from data_representations import BOW

class Preprocessor():
    def __init__(self, filepath, keep_punc=False):
        """Preprocessing class, does reading and tokenising.

        Args:
            filename (string): path to the data set file
            keep_punc (bool, optional): Keep punctuation in the tokens-option. Defaults to False.
        """
        self.keep_punc = keep_punc
        self.artists, self.titles, self.tokenized_lyrics = self.read(filepath)

    def read(self, filepath):
        """Read file and tokenize lyrics 

        Args:
            filename (string): path to the data set file

        Returns:
            list(string), list(string), list(list(string)): list of artists, list of songtitles, list of tokens
        """
        artists, titles, tokenized_lyrics = [], [], []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                # The three values for each line are joined together with the tab '\t' character
                artist, title, lyrics = line.split('\t')

                tokenized = self.tokenize(lyrics) 

                artists.append(artist)
                titles.append(title)
                tokenized_lyrics.append(tokenized)

        return artists, titles, tokenized_lyrics

    def tokenize(self, lyrics):
        """Tokenize

        Args:
            lyrics (string): Lyrics of a song to be tokenized

        Returns:
            list(string): list of words in the lyrics
        """
        if not self.keep_punc:
            return re.sub(r'[^\w\s]', '', lyrics.replace(" NEWLINE ", ' ').replace(" NEWLINE\n", '')).split(' ')
        else:
            return lyrics.replace(" NEWLINE ", ' ').replace(" NEWLINE\n", '').split(' ')

    def BOW(self):
        """Getting a BOW representation of the lyrics

        Returns:
            list(BOW): list of BOW per lyrics of song in the data set
        """
        return [BOW(text) for text in self.tokenized_lyrics]

    def artists(self): 
        """Alias for preprocessor.artists

        Returns:
            list(string): list of artists of the data set
        """
        return self.artists

    def titles(self):
        """Alias for preprocessor.titles

        Returns:
            list(string): list of titles of the data set
        """
        return self.titles

if __name__ == "__main__":
    pre = Preprocessor("/home/mmm/Dropbox/Org/Uni/SS22/teamlab2022/data/songs_train.txt")
    bow = pre.BOW()
    print(bow[0].similarity(bow[1]))
    print(len(pre.artists) == len(pre.titles) == len(pre.tokenized_lyrics))

