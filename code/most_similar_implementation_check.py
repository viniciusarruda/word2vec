import fire
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors
import gensim.downloader as api
from word_vectors import WordVectors
import math


def load_analogies(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        analogies_lines = f.readlines()
    analogies_lines = [line.strip() for line in analogies_lines]
    analogies_lines = [line for line in analogies_lines if not line.startswith("//") and not line.startswith(": ")]
    analogies_lines = [line.split() for line in analogies_lines if len(line) > 0]

    assert all(len(line) == 4 for line in analogies_lines)

    return analogies_lines


def load_models(model: str):
    if model == "word2vec-google-news-300":
        # loading gensim model
        gensim_model = api.load("word2vec-google-news-300")

        # Option to load a downloaded file limiting the size
        # gensim_model = KeyedVectors.load_word2vec_format(
        #     "../dataset/GoogleNews-vectors-negative300.bin", binary=True, limit=100000
        # )

        # loading original google word2vec in gensim and convert to my format to use my most_similar implementation
        this_model = WordVectors(
            index_to_key=gensim_model.index_to_key,
            vectors=np.array([gensim_model[w] for w in gensim_model.index_to_key], dtype=np.float64),
        )

    else:
        this_model = WordVectors.from_file(model)
        gensim_model = KeyedVectors.load_word2vec_format(model, binary=False)

    return gensim_model, this_model


def main(model_path_or_name: str) -> None:
    """Assert the result for most_similar from gensim is the same as this most_similar implementation.
    The assertion is done by comparing the output of each implementations when computing the analogies in file ../dataset/word-test.v1.txt.

    Args:
        model_path_or_name (str): Path for the trained model (a txt file containing the word vectors) or the model name for a model from gensim, for example: word2vec-google-news-300 to use the original google word2vec embeddings.
    """
    analogies = load_analogies("../dataset/word-test.v1.txt")

    gensim_model, this_model = load_models(model_path_or_name)

    assert gensim_model.key_to_index == this_model.key_to_index

    oov, not_oov = 0, 0
    for analogy in tqdm(analogies):
        # vocab is same for both my and gensim model
        if not all(e in this_model.key_to_index for e in analogy):
            oov += 1
            continue
        not_oov += 1

        # x1 is to x2 as y1 is to y2
        x1, x2, y1, y2 = analogy
        # expected: x2 - x1 + y1 ~ y2

        gensim_most_similar = gensim_model.most_similar(positive=[x2, y1], negative=[x1], topn=5)
        this_most_similar = this_model.most_similar(positive=[x2, y1], negative=[x1], topn=5)

        assert all(
            g[0] == m[0] and math.isclose(g[1], m[1], abs_tol=0.000001) for g, m in zip(gensim_most_similar, this_most_similar)
        )

    print("OOV: ", oov)
    print("NOT OOV: ", not_oov)


if __name__ == "__main__":
    fire.Fire(main)
