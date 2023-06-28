import os
import json
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px


def pca(data, words, file_name, labels, n_components=2):
    # TODO use the sklearn version
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    evals, evecs = linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :n_components]
    result = np.dot(evecs.T, data.T).T
    _plot(result, words, file_name, labels)


def tsne(data, words, file_name, labels):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    result = tsne.fit_transform(data)
    _plot(result, words, file_name, labels)


def _plot(data, words, file_name, labels=None):
    fig = plt.figure(figsize=(10, 10))

    if labels is None:
        plt.scatter(data[:, 0], data[:, 1])
    else:
        plt.scatter(data[:, 0], data[:, 1], c=labels)

    for i, word in enumerate(words):
        plt.annotate(word, xy=(data[i, 0], data[i, 1]))

    plt.tight_layout()
    fig.savefig(file_name, dpi=fig.dpi, bbox_inches="tight")
    plt.close()


def plot_embeddings(
    folder_path: str,
    unformatted_file_path: str,
    embeddings: np.ndarray,
    words: list[int],
    idxs: list[int] = None,
    labels: list[int] = None,
):
    if idxs is not None:
        embeddings = embeddings[idxs, :]

    pca(embeddings, words, os.path.join(folder_path, unformatted_file_path.format(plot_type="pca")), labels)
    tsne(embeddings, words, os.path.join(folder_path, unformatted_file_path.format(plot_type="tsne")), labels)


def plot_embeddings_from_file(
    file_path: str, save_folder_path: str, save_unformatted_file_path: str, embeddings: np.ndarray, vocab
):
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data = vocab.filter(data, lower=True)

        keys = sorted(data[0].keys())
        words, labels = [], []
        for entry in data:
            words += [entry[k] for k in keys]
            labels += list(range(len(keys)))
    else:
        assert file_path.endswith(".txt")
        with open(file_path, "r", encoding="utf-8") as f:
            words = f.readlines()
        words = [l.strip() for l in words]
        words = [l for l in words if len(l) > 0]
        words = vocab.filter(words, lower=True)[:1000]
        labels = None

    plot_embeddings(
        save_folder_path,
        save_unformatted_file_path,
        embeddings,
        words,
        idxs=vocab.get_idxs(words),
        labels=labels,
    )


def plot_word_freq(words: list[str], save_file_path: str):
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    freq = sorted(freq.items(), key=lambda x: x[1])

    fig = px.bar(freq, x=1, y=0, labels={"0": "word", "1": "frequency"})
    fig.write_html(save_file_path, include_plotlyjs="cdn")
