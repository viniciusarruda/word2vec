import fire
import numpy as np
from tqdm import tqdm
import gensim.downloader as api
from word_vectors import WordVectors
from prettytable import PrettyTable, MARKDOWN


def load_analogies(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        analogies_lines = f.readlines()
    analogies_lines = analogies_lines[1:]  # drop copyright comment
    analogies_lines = [line.strip() for line in analogies_lines]

    analogies = {}
    for line in analogies_lines:
        line = line.strip()
        if len(line) > 0:
            if line.startswith(": "):
                current_class = line.replace(": ", "")
                analogies[current_class] = []
            else:
                analogies[current_class].append(line.split())
                assert len(analogies[current_class][-1]) == 4

    # double-check
    assert len(analogies) == 14
    # 19559 - 14 - 1 = 19544 # num of analogies
    assert sum(len(analogies_per_class) for analogies_per_class in analogies.values()) == 19544

    return analogies


def dict2table(data: dict) -> str:
    table = PrettyTable(["Analogy Class", "OOV", "not OOV", "Top1", "Top5", "Total"])
    table.set_style(MARKDOWN)

    total = {"OOV": 0, "Top1": 0, "Top5": 0, "Total": 0}
    for analogy_class, analysis in data.items():
        not_oov = analysis["Total"] - analysis["OOV"]
        table.add_row(
            [
                analogy_class,
                f'{analysis["OOV"]} ({analysis["OOV"] / analysis["Total"]:.2%})',
                f'{not_oov} ({not_oov / analysis["Total"]:.2%})',
                f'{analysis["Top1"]} ({analysis["Top1"] / analysis["Total"]:.2%})',
                f'{analysis["Top5"]} ({analysis["Top5"] / analysis["Total"]:.2%})',
                analysis["Total"],
            ]
        )
        for k, v in analysis.items():
            total[k] += v

    # table._dividers[-1] = True
    # github seems to not like this last divider, thus use **bold** for the total row
    not_oov = total["Total"] - total["OOV"]
    table.add_row(
        [
            "**Total**",
            f'**{total["OOV"]}** (**{total["OOV"] / total["Total"]:.2%}**)',
            f'**{not_oov}** (**{not_oov / total["Total"]:.2%}**)',
            f'**{total["Top1"]}** (**{total["Top1"] / total["Total"]:.2%}**)',
            f'**{total["Top5"]}** (**{total["Top5"] / total["Total"]:.2%}**)',
            f'**{total["Total"]}**',
        ]
    )
    return table.get_string()


def main(model_path_or_name: str, output_path: str) -> None:
    """Compute the accuracy of the analogies in file ../dataset/word-test.v1.txt

    Args:
        model_path_or_name (str): Path for the trained model (a txt file containing the word vectors) or the model name for a model from gensim, for example: word2vec-google-news-300 to use the original google word2vec embeddings.
        output_path (str): Path for the file to save the results.
    """
    analogies = load_analogies("../dataset/word-test.v1.txt")

    if model_path_or_name == "word2vec-google-news-300":
        gensim_model = api.load("word2vec-google-news-300")

        model = WordVectors(
            index_to_key=gensim_model.index_to_key,
            vectors=np.array([gensim_model[w] for w in gensim_model.index_to_key], dtype=np.float64),
        )
    else:
        model = WordVectors.from_file(model_path_or_name)

    analysis = {}
    for analogy_class, analogies_per_class in tqdm(analogies.items(), desc="Comparing analogies"):
        analysis[analogy_class] = {"OOV": 0, "Top1": 0, "Top5": 0, "Total": len(analogies_per_class)}
        for analogy in tqdm(analogies_per_class, leave=False, desc=f"analogy class: {analogy_class}"):
            # vocab is same for both my and gensim model
            if not all(e in model.key_to_index for e in analogy):
                analysis[analogy_class]["OOV"] += 1
                continue

            # x1 is to x2 as y1 is to y2
            x1, x2, y1, y2 = analogy
            # expected: x2 - x1 + y1 ~ y2

            top_most_similar = model.most_similar(positive=[x2, y1], negative=[x1], topn=5)

            if top_most_similar[0][0] == y2:
                analysis[analogy_class]["Top1"] += 1

            if y2 in list(zip(*top_most_similar))[0]:
                analysis[analogy_class]["Top5"] += 1

    table = dict2table(analysis)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table)


if __name__ == "__main__":
    fire.Fire(main)
