import fire
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors
import gensim.downloader as api
from word_vectors import WordVectors
import math
from prettytable import PrettyTable, MARKDOWN

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


def summary(self, embeddings, vocab: dict[str, int], special_tokens: set[str] | None = None) -> tuple[float, float, str]:
    if special_tokens is not None:
        assert all(st in vocab for st in special_tokens)
        special_token_idxs = [vocab[st] for st in special_tokens]
        embeddings = np.delete(embeddings, special_token_idxs, axis=0)
        new_vocab = dict()
        offset = 0
        for k, v in vocab.items():
            if k in special_tokens:
                offset += 1
                continue
            new_vocab[k] = v - offset
        vocab = new_vocab

    # if padding_token:
    #     # excluding <pad> embedding with id 0
    #     embeddings = embeddings[1:, :]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms

    s = {}
    for line in tqdm(self.analogy_lines, desc="Computing vector analogy", leave=False):
        if line.startswith("//"):
            continue
        elif line.startswith(": "):
            class_name = line.replace(": ", "")
            assert class_name not in s
            s[class_name] = {"top1": 0, "top5": 0, "total": 0}
        else:
            top1, top5 = self._most_similar(embeddings_norm, vocab, line.split())
            s[class_name]["top1"] += top1
            s[class_name]["top5"] += top5
            s[class_name]["total"] += 1

    table = PrettyTable(["Class name", "Top1", "Top5", "Total"])
    table.set_style(MARKDOWN)
    top1, top5, total = 0, 0, 0
    for class_name in s:
        top1 += s[class_name]["top1"]
        top5 += s[class_name]["top5"]
        total += s[class_name]["total"]
        table.add_row(
            [
                class_name,
                f"{s[class_name]['top1']} ({s[class_name]['top1'] / s[class_name]['total']:.2%})",
                f"{s[class_name]['top5']} ({s[class_name]['top5'] / s[class_name]['total']:.2%})",
                s[class_name]["total"],
            ]
        )
    table._dividers[-1] = True
    table.add_row(
        [
            "Total",
            f"{top1} ({top1 / total:.2%})",
            f"{top5} ({top5 / total:.2%})",
            total,
        ]
    )

    return top1 / total, top5 / total, table.get_string()


def main(model_path: str) -> None:
    analogies = load_analogies("../dataset/word-test.v1.txt")

    model = WordVectors.from_file(model_path)

    oov, not_oov = 0, 0
    for analogy in tqdm(analogies):
        # vocab is same for both my and gensim model
        if not all(e in model.key_to_index for e in analogy):
            oov += 1
            continue
        not_oov += 1

        # x1 is to x2 as y1 is to y2
        x1, x2, y1, _ = analogy
        # expected: x2 - x1 + y1 ~ y2

        this_most_similar = model.most_similar(positive=[x2, y1], negative=[x1], topn=5)

    print("OOV: ", oov)
    print("NOT OOV: ", not_oov)


if __name__ == "__main__":
    fire.Fire(main)
