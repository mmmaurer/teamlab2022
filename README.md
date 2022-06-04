# Artist classification
This is the repository of the project for the Team Laboratory Computational Linguistics course at the University of Stuttgart in the summer term of 2022. 

The artist classification task is to classify which artist performed/wrote a certain song given the lyrics.

## Dataset

Per instance (song), the dataset includes a triple with the *artist name*, *song title* and the *lyrics* of a song. 

Overview over the number of artists/songs in the dataset:
- | | Train | Val | Test |
    --- | --- | --- | --- |
    Artists | 642 | 612 | 618 |
    Songs | 46,120 | 5,765| 5,765|

## Baseline implementation
TODO: short description about the baseline implementation (just overview of all the headings in the document for this)

The baselines of this are a [k-nearest neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) classifier ([implementation](./src/classifiers/knn.py)).

We experiment with different combinations of data representations and distance/similarity measures as well as with additional stylometric features (number of lines in lyric, length of lyric in words).

### Evaluations
We evaluate our experiments using Accuracy as well as micro-averaged Precision, Recall and F<sub>1</sub>-Score. Find these and the implementations macro-averaged metrics [here](./src/evaluation/evaluation.py).


### Data representations
We use two kinds of representations for the lyrics
- [Bags of Words (BOW)](https://en.wikipedia.org/wiki/Bag-of-words_model) implemented as sets ([implementation](./src/data_representations/bow.py))

    Distance/similarity measures:
    - [Tversky index](https://en.wikipedia.org/wiki/Tversky_index)
    - [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)
    - [Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
    - [Overlap coefficient](https://en.wikipedia.org/wiki/Overlap_coefficient)

- [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vectors ([implementation](./src/data_representations/tf_idf.py))

    Distance/similarity measures:
    - [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
    - [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)

### Results
Find the experiments and results in these two notebooks: 
- [Experiments 1](./notebooks/experiments.ipynb)
- [Experiments 2](./notebooks/experiments_server.ipynb)

|                      |                            | 10k train |           | 20k train |           |
|----------------------|----------------------------|-----------|-----------|-----------|-----------|
|                      | Distance/similarity metric | Acc       | Micro-F<sub>1</sub> | Acc       | Micro-F<sub>1</sub> |
| Random choice        |                            | 0.004     | 0.007     | 0.001     | 0.002     |
| BOW                  | Jaccard                    | **0.07**  | **0.115** | **0.09**  | **0.149** |
|                      | Sørensen-Dice              | **0.07**  | **0.115** | **0.09**  | **0.149** |
|                      | Tversky                    | 0.06      | 0.094     | 0.05      | 0.083     |
| TF-IDF               | Cosine                     | **0.07**      | 0.108     | 0.08      | 0.125     |
|                      | Euclidean                  | 0.01      | 0.018     | 0.04      | 0.071     |
| TF-IDK + Stylometric features| Cosine                     | 0.02      | 0.032     | 0.01      | 0.017     |
|                      | Euclidean                  | 0.01      | 0.017     | 0.01      | 0.018     |