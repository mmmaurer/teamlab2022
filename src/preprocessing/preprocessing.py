import re


class Preprocessor():
    def __init__(self, filepath, keep_punc=False, read_limit=1000):
        """Preprocessing class, does reading and tokenising.

        Args:
            filename (string): path to the data set file
            keep_punc (bool, optional): Keep punctuation in the tokens-option.
                                        Defaults to False.
            read_limit (int, optional): number of examples to read from file.
                                        Defaults to 1000.
        """
        self.keep_punc = keep_punc
        self.artists, self.titles, self.tokenized = self.read(filepath,
                                                              read_limit)

    def read(self, filepath, read_limit=1000):
        """Read file and tokenize lyrics

        Args:
            filename (string): path to the data set file
            read_limit (int, optional): number of examples to read from file.
                                        Defaults to 1000.

        Returns:
            list(string), list(string), list(list(string)): list of artists,
                                                            list of songtitles,
                                                            list of tokens
        """
        artists, titles, tokenized_lyrics = [], [], []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i >= read_limit:
                    break

                # The three values for each line are joined together with the
                # tab '\t' character
                artist, title, lyrics = line.split('\t')

                tokenized = self.tokenize(lyrics)

                artists.append(artist)
                titles.append(title)
                tokenized_lyrics.append(tokenized)

        return artists, titles, tokenized_lyrics

    def tokenize(self, lyrics):
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
        if not self.keep_punc:  # remove punctuation
            return re.sub(r'[^\w\s]', '', cleaned).split(' ')
        else:
            return cleaned.split(' ')
