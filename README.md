<a name="readme-top"></a>

<br />
<div align="center">
  <a href="https://github.com/viniciusarruda/word2vec/">
    <img src="images/word2vec.png" width="500">
  </a>
  <p align="center" >Source: <a href="http://jalammar.github.io/illustrated-word2vec/">The Illustrated Word2vec</a></p>
  <h3 align="center">Yet Another Word2Vec Implementation</h3>
</div>

## About

Implementation of [Continuous Bag-of-Words](https://arxiv.org/abs/1301.3781) (CBOW) in pytorch.

Features:

- Train a CBOW from scratch
- Log training to tensorboard
- Visualize embeddings with t-SNE/PCA/UMAP using tensorboard.
- Implements a `most_similar` function with the same behavior and results of the [`most_similar`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar) function implemented by the [`Gensim`](https://radimrehurek.com/gensim/) library.

## Installation

Note: 
> This project was developed using `Windows 11` with `python 3.10.0`.


Clone this repo, create a new environment (recommended) and install the dependencies:

```bash
pip install -r requirements.txt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Train a CBOW model

Download the dataset `WikiText-2` or `WikiText-103` [here](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/#download) and move it into the `dataset` folder.

Edit the `config.toml` accordingly, then:
```bash
python main.py
```

To use tensorboard (setting scalars to [show all datapoints](https://stackoverflow.com/questions/43702546/tensorboard-doesnt-show-all-data-points)):
```bash
tensorboard --logdir .\experiment\wikitext-2\ --samples_per_plugin scalars=300000
```

The dataset included in this repo can be downloaded [here](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/#download).

### Compute analogies

To compute the analogies and summarize them using the `word-test.v1.txt`, the original test set [file](www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt) from the word2vec paper.

To run the original trained word2vec (it will download the model):
```bash
python compute_analogies.py word2vec-google-news-300
```
The results from the above script can be seen [below](#word2vec-google-news-300).

To run with a trained word2vec, use the path from a `txt` file containing the word vectors:
```bash
python compute_analogies.py <path-to-txt-word-vectors>
```

### Checking most_similar implementation

[`most_similar`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar) is a function from the [`Gensim`](https://radimrehurek.com/gensim/) which retrieves the top-N most similar embeddings. The goal of the `most_similar_implementation_check.py` script is to assert the equality of results between `most_similar` implementation. 

To run the original trained word2vec (it will download the model):
```bash
python most_similar_implementation_check.py word2vec-google-news-300
```

Or use the path from a `txt` file containing the word vectors:
```bash
python most_similar_implementation_check.py <path-to-txt-word-vectors>
```

## Results

### word2vec-google-news-300

|        Analogy Class        |        OOV        |         not OOV         |          Top1          |          Top5          |   Total   |
|:---------------------------:|:-----------------:|:-----------------------:|:----------------------:|:----------------------:|:---------:|
|   capital-common-countries  |     0 (0.00%)     |      506 (100.00%)      |      421 (83.20%)      |      482 (95.26%)      |    506    |
|        capital-world        |     0 (0.00%)     |      4524 (100.00%)     |     3580 (79.13%)      |     4124 (91.16%)      |    4524   |
|           currency          |     0 (0.00%)     |      866 (100.00%)      |      304 (35.10%)      |      431 (49.77%)      |    866    |
|        city-in-state        |     0 (0.00%)     |      2467 (100.00%)     |     1749 (70.90%)      |     2127 (86.22%)      |    2467   |
|            family           |     0 (0.00%)     |      506 (100.00%)      |      428 (84.58%)      |      482 (95.26%)      |    506    |
|  gram1-adjective-to-adverb  |     0 (0.00%)     |      992 (100.00%)      |      283 (28.53%)      |      509 (51.31%)      |    992    |
|        gram2-opposite       |     0 (0.00%)     |      812 (100.00%)      |      347 (42.73%)      |      457 (56.28%)      |    812    |
|      gram3-comparative      |     0 (0.00%)     |      1332 (100.00%)     |     1210 (90.84%)      |     1295 (97.22%)      |    1332   |
|      gram4-superlative      |     0 (0.00%)     |      1122 (100.00%)     |      980 (87.34%)      |     1102 (98.22%)      |    1122   |
|   gram5-present-participle  |     0 (0.00%)     |      1056 (100.00%)     |      825 (78.12%)      |     1004 (95.08%)      |    1056   |
| gram6-nationality-adjective |     0 (0.00%)     |      1599 (100.00%)     |     1438 (89.93%)      |     1527 (95.50%)      |    1599   |
|       gram7-past-tense      |     0 (0.00%)     |      1560 (100.00%)     |     1029 (65.96%)      |     1459 (93.53%)      |    1560   |
|         gram8-plural        |     0 (0.00%)     |      1332 (100.00%)     |     1197 (89.86%)      |     1275 (95.72%)      |    1332   |
|      gram9-plural-verbs     |     0 (0.00%)     |      870 (100.00%)      |      591 (67.93%)      |      785 (90.23%)      |    870    |
|          **Total**          | **0** (**0.00%**) | **19544** (**100.00%**) | **14382** (**73.59%**) | **17059** (**87.29%**) | **19544** |


## Resources

- Links regarding the `most_similar` and analogy computation: [1](https://blog.esciencecenter.nl/king-man-woman-king-9a7fd2935a85), [2](https://stackoverflow.com/questions/54580260/understanding-gensim-word2vecs-most-similar), [3](https://stackoverflow.com/questions/59590993/where-can-i-download-a-pretrained-word2vec-map), [4](https://stackoverflow.com/questions/65059959/gensim-most-similar-with-positive-and-negative-how-does-it-work).
- Tensorboard: [1](https://cnvrg.io/tensorboard-guide/), [2](https://www.machinelearningnuggets.com/tensorboard-tutorial/), [3](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html).