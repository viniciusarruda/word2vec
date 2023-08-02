import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors
import gensim.downloader as api
from word_vectors import WordVectors
import math

# # TODO
# bring summary here
# - make a split in if main entry to select to compare most_similar implementation (already done here)
# - run models and build a table for comparison, e.g, mine run with google one, reporting top1 and top5 (see summary)

# - check codes, clean, run evaluations, update repo

# until here, can be done today, in 3 hours!!!!

# then, study tomorrow hierarchical softmax and if feasible implement it here
# also, see negative sampling, which I think it will work better.

# this week you finish this repo! and then check the study plan, but only if relevant, because you need to go to transformers stuff and andrej stuff.


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
        my_model = WordVectors(
            index_to_key=gensim_model.index_to_key,
            vectors=np.array([gensim_model[w] for w in gensim_model.index_to_key], dtype=np.float64),
        )

    else:
        my_model = WordVectors.from_file(model)

        gensim_model = KeyedVectors.load_word2vec_format(model, binary=False)

    return gensim_model, my_model


def main():
    analogies = load_analogies("../dataset/word-test.v1.txt")

    # Important to test using a model done by gensim
    gensim_model, my_model = load_models("word2vec-google-news-300")

    # Important to test my saving and loading method with gensim
    # gensim_model, my_model = load_models("../experiment/nltk-brown/CBOW/12-07-2023_16-03-39/model/word_embeddings.txt")
    # gensim_model, my_model = load_models("w2v_test.txt")

    assert gensim_model.key_to_index == my_model.key_to_index

    oov, not_oov = 0, 0
    for analogy in tqdm(analogies):
        # vocab is same for both my and gensim model
        if not all(e in my_model.key_to_index for e in analogy):
            oov += 1
            continue
        not_oov += 1

        # x1 is to x2 as y1 is to y2
        x1, x2, y1, y2 = analogy
        # expected: x2 - x1 + y1 ~ y2

        gensim_most_similar = gensim_model.most_similar(positive=[x2, y1], negative=[x1], topn=5)

        my_most_similar = my_model.most_similar(positive=[x2, y1], negative=[x1], topn=5)

        assert all(
            g[0] == m[0] and math.isclose(g[1], m[1], abs_tol=0.000001) for g, m in zip(gensim_most_similar, my_most_similar)
        )

        # for g, m in zip(gensim_most_similar, my_most_similar):
        #     if g[0] != m[0] or not math.isclose(g[1], m[1], abs_tol=0.000001):
        #         print("------")
        #         print(analogy)
        #         print("gensim_most_similar", gensim_most_similar)
        #         print("my_most_similar", my_most_similar)
        #         break
        #         # exit()

    print("OOV: ", oov)
    print("NOT OOV: ", not_oov)


if __name__ == "__main__":
    main()
