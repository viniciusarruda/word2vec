import numpy as np


class WordVectors:
    def __init__(self, index_to_key: list[str], vectors: np.ndarray, special_tokens: set[str] | None = None) -> None:
        self.special_tokens = special_tokens
        self.vectors = vectors
        self.index_to_key = index_to_key
        self.key_to_index = {w: i for i, w in enumerate(self.index_to_key)}

        if self.special_tokens is not None:
            assert self.vectors is not None
            self.index_to_key, self.key_to_index, self.vectors = self._remove_special_tokens(
                self.index_to_key, self.key_to_index, self.special_tokens, self.vectors
            )
            self.special_tokens = None

        self.vectors = self._normalize_vectors(self.vectors)

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    @staticmethod
    def _remove_special_tokens(
        index_to_key: list[str], key_to_index: dict[str, int], special_tokens: set[str], vectors: np.ndarray
    ) -> tuple[list[str], dict[str, int], np.ndarray]:
        assert all(st in key_to_index for st in special_tokens)

        special_token_idxs = [key_to_index[st] for st in special_tokens]

        vectors = np.delete(vectors, special_token_idxs, axis=0)

        for i in special_token_idxs:
            del index_to_key[i]

        key_to_index = {w: i for i, w in enumerate(index_to_key)}

        return index_to_key, key_to_index, vectors

    # TODO ensure I'm removing special tokens. Don't save with special tokens. Don't load special tokens?
    @staticmethod
    def save_vectors(
        index_to_key: list[str],
        key_to_index: dict[str, int],
        special_tokens: set[str] | None,
        vectors: np.ndarray,
        filepath: str,
    ) -> None:
        if special_tokens is not None:
            index_to_key, key_to_index, vectors = WordVectors._remove_special_tokens(
                index_to_key, key_to_index, special_tokens, vectors
            )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{len(index_to_key)} {vectors.shape[1]}")

            for i, w in enumerate(index_to_key):
                if w not in special_tokens:
                    f.write(f"\n{w} ")
                    np.savetxt(f, vectors[i, np.newaxis], fmt="%.12f", newline="")

    @classmethod
    def from_file(cls: "WordVectors", filepath: str) -> "WordVectors":
        with open(filepath, "r", encoding="utf-8") as f:
            data_lines = f.readlines()

        index_to_key, vectors = [], []
        for line in data_lines[1:]:
            w, e = line.strip().split(maxsplit=1)
            vectors.append(np.fromstring(e, sep=" "))
            index_to_key.append(w)
        vectors = np.stack(vectors, axis=0)

        return cls(index_to_key=index_to_key, vectors=vectors)

    # kernprof -l .\compare_most_similar_implementation.py
    # python -m line_profiler .\compare_most_similar_implementation.py.lprof
    # @profile
    def most_similar(self, positive: list[str], negative: list[str], topn: int = 5) -> list[tuple[str, float]]:
        if len(positive) == 0 or len(negative) == 0:
            raise NotImplementedError("Not implemented to receive empty positive or empty negative.")

        positive_idxs = [self.key_to_index[key] for key in positive]
        negative_idxs = [self.key_to_index[key] for key in negative]
        input_idxs = positive_idxs + negative_idxs

        weights = np.ones((len(input_idxs), 1), dtype=self.vectors.dtype)
        weights[len(positive_idxs) :] *= -1

        mean_vector = np.mean(self.vectors[input_idxs] * weights, axis=0)
        mean_vector /= np.linalg.norm(mean_vector)

        dists = np.matmul(self.vectors, mean_vector)

        topn_offset = -topn - len(input_idxs)
        topn_idxs = np.argpartition(dists, topn_offset)[topn_offset:]
        topn_idxs_sorted = np.argsort(-dists[topn_idxs])
        topn_idxs = topn_idxs[topn_idxs_sorted]

        # filter input vectors as gensim most_similar implementation (tricky isn't it?)
        topn_idxs = [i for i in topn_idxs if i not in input_idxs][:topn]

        return [(self.index_to_key[idx], dists[idx]) for idx in topn_idxs]
