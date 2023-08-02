import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import time
import itertools


class Timelapse:
    def __init__(self, name):
        self.start = None
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print(f"Begin {self.name}..")
        return self

    def __exit__(self, type, value, traceback):
        print(f"End {self.name} (took {time.time() - self.start:.2f} seconds).")


def load_processed_file_data(file_path: str) -> list[list[str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.readlines()

    data = [line.strip() for line in data]
    data = [line for line in data if len(line) > 0]
    data = [line.split() for line in data]

    assert all([len(d) > 0 for d in data])

    return data


class Vocabulary:
    def __init__(
        self, data: list[list[str]] = None, folder_path: str = None, min_freq: int = 1, default_token: str = "<unk>"
    ) -> None:
        if data is not None:
            vocab = self.build_vocab(data, min_freq)
            assert type(vocab) is set
            assert "<pad>" not in vocab

            if default_token in vocab:
                print(f"WARNING: Default index ({default_token}) found in data")
                vocab.remove(default_token)

            vocab = ["<pad>", default_token] + sorted(vocab)  # pad must be at index 0!
            self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
            self.idx_to_word = {idx: word for idx, word in enumerate(vocab)}  # TODO make as a list, save as a list ?

            if folder_path is not None:
                Vocabulary._save_vocab_map(self.word_to_idx, os.path.join(folder_path, "word_to_idx.json"))
                Vocabulary._save_vocab_map(self.idx_to_word, os.path.join(folder_path, "idx_to_word.json"))
        else:
            # TODO test this, or delete if you wont use
            assert folder_path is not None
            self.word_to_idx = Vocabulary._load_vocab_map(os.path.join(folder_path, "word_to_idx.json"))
            self.idx_to_word = Vocabulary._load_vocab_map(os.path.join(folder_path, "idx_to_word.json"))
            assert default_token in self.word_to_idx
            assert (
                self.word_to_idx[default_token] in self.idx_to_word
                and self.idx_to_word[self.word_to_idx[default_token]] == default_token
            )
        self.default_token = default_token
        self.default_idx = self.word_to_idx[self.default_token]

    def __len__(self) -> int:
        return len(self.word_to_idx)

    def stoi(self, data: list[list[str]]) -> list[list[int]]:
        return [[self.word_to_idx.get(word, self.default_idx) for word in line] for line in data]

    def itos(self, idxs: list[int] | None = None) -> list[str]:
        if idxs is None:
            idxs = range(len(self))
        return [self.idx_to_word[idx] for idx in idxs]

    def save_data_stats(self, data: list[list[str]], filename: str, folder_path: str) -> None:
        data = list(itertools.chain.from_iterable(data))
        n = len(data)
        data = [word if word in self.word_to_idx else self.default_token for word in data]
        assert n == len(data)  # Duh!
        data_freq = Counter(data).most_common()
        stats = "\n".join([f"{word}\t{freq}\t{freq/n:.8f}" for word, freq in data_freq])
        file_path = os.path.join(folder_path, f"{filename}.tsv")
        assert not os.path.exists(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(stats)

    @staticmethod
    def build_vocab(data: list[list[str]], min_freq: int) -> set:
        data_freq = Counter(itertools.chain.from_iterable(data))
        return {token for token, freq in data_freq.items() if freq >= min_freq}

    @staticmethod
    def _load_vocab_map(file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @staticmethod
    def _save_vocab_map(data: dict, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


class ContextDataset(Dataset):
    # skip unknown words? they help to un-learn? haha
    def __init__(self, data_as_idxs: list[list[int]], context_size: int):
        self.context_size = context_size
        self.data_as_idxs = data_as_idxs
        self.length = sum((len(sent) for sent in self.data_as_idxs))
        self._build_idx_to_item()

        assert self.length == self.cummulative_length[-1]

    def _build_idx_to_item(self):
        self.sentence_idxs, self.cummulative_length = [], [0]
        for i, sentence in enumerate(self.data_as_idxs):
            self.sentence_idxs += [i] * len(sentence)
            self.cummulative_length.append(self.cummulative_length[-1] + len(sentence))

    # TODO need to debug this to check if is correct
    # continuar fazendo esse w2v funcionar!
    # vc ta tao parado nas suas ideias doidas que esqueceu de fazer o negocio funcionar, para de inventar a roda e vai para o relevante!
    # ver o lance do padding idx (comparar com embbed sum e mean) -> acho que ja vi alguem usando isso, ver suas tabs!
    # debugar e ver se esta implementado certo aqui
    # ver eficacia das coisas
    # ir para hierarchical softmax
    # dps negative sampling
    # depois acabou tudo de word2vec
    # ai ir para ajeitar o codigo (vai virar um boilerplate)
    # e pronto!

    def __len__(self) -> int:
        # return len(self.data_as_idxs) - 2 * self.context_size
        return self.length

    def __getitem__(self, idx):
        sentence_idx = self.sentence_idxs[idx]
        idx_offset = self.cummulative_length[sentence_idx]
        sentence = self.data_as_idxs[sentence_idx]

        idx -= idx_offset
        assert idx >= 0

        left_context = max(self.context_size - idx, 0) * [0] + sentence[max(idx - self.context_size, 0) : idx]

        center_idx = sentence[idx]

        right_context = sentence[idx + 1 : idx + self.context_size + 1] + max(
            (idx + self.context_size + 1) - len(sentence), 0
        ) * [0]

        # logica para right pad
        # dps rever logica do left pad
        # min(idx + self.context_size + 1, len(sentence)) - len(sentence)
        # max(idx + self.context_size + 1, len(sentence)) - len(sentence)

        # max(len(sentence) - self.context_size - 1, 0)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # if len(sentence) - self.context_size - 1 < idx:
        #     precisa de pad

        context_idxs = left_context + right_context

        assert len(context_idxs) == 2 * self.context_size

        context_input = torch.tensor(context_idxs, dtype=torch.long)

        return context_input, center_idx

    # def __getitem__(self, idx):
    #     idx += self.context_size
    #     left_context = self.data_as_idxs[idx - self.context_size : idx]
    #     center_idx = self.data_as_idxs[idx]
    #     right_context = self.data_as_idxs[idx + 1 : idx + self.context_size + 1]
    #     context_idxs = left_context + right_context

    #     context_input = torch.tensor(context_idxs, dtype=torch.long)

    #     return context_input, center_idx


def get_processed_data(context_size: int, dataset_config: dict, dataloader_config: dict, output_folder_path: str) -> dict:
    # TODO colocar docstring automaticamente
    dataset_names = dataset_config.keys()

    stats_folder_path = os.path.join(output_folder_path, "stats")
    os.makedirs(stats_folder_path)

    vocab_folder_path = os.path.join(output_folder_path, "vocab")

    datasets = {}
    if "train" in dataset_names:
        os.makedirs(vocab_folder_path)
        train_data = load_processed_file_data(dataset_config["train"]["file_path"])

        with Timelapse("Vocabulary"):
            vocab = Vocabulary(data=train_data, folder_path=vocab_folder_path, min_freq=dataset_config["train"]["min_freq"])

        with Timelapse("Saving vocab"):
            vocab.save_data_stats(train_data, "train", stats_folder_path)

        with Timelapse("stoi"):
            train_data_as_idxs = vocab.stoi(train_data)

        with Timelapse("Getting training dataset"):
            datasets["train"] = ContextDataset(train_data_as_idxs, context_size)
    else:
        # TODO need to test this
        vocab = Vocabulary(folder_path=vocab_folder_path)

    for name in dataset_names:
        if name != "train":  # already processed train
            data = load_processed_file_data(dataset_config[name]["file_path"])

            vocab.save_data_stats(data, name, stats_folder_path)
            data_as_idxs = vocab.stoi(data)
            datasets[name] = ContextDataset(data_as_idxs, context_size)

    dataloaders = {}
    for name in dataset_names:
        print(name, dataloader_config[name])
        dataloaders[name] = DataLoader(datasets[name], **dataloader_config[name])

    return dataloaders, vocab


if __name__ == "__main__":
    from tqdm import tqdm

    # data = [["This", "is", "a", "test", "!"]]

    data = load_processed_file_data("../dataset/wikitext-103/wiki.train.tokens")

    vocab = Vocabulary(data=data, min_freq=1)

    print("--- Vocab ---")
    # print(vocab.word_to_idx)
    # print(vocab.idx_to_word)
    print(vocab.idx_to_word[0])
    print(vocab.idx_to_word[1])
    print(vocab.default_token)
    print(vocab.default_idx)
    print(len(vocab))
    print("-------------")

    data_as_idxs = vocab.stoi(data)
    del data
    print("--- data_as_idxs ---")
    # print(data_as_idxs[:10])
    print("-------------")

    dataset = ContextDataset(data_as_idxs, context_size=5)
    print("--- dataset ---")
    print(len(dataset))
    # for i, item in enumerate(dataset):
    #     print(item)
    #     if i == 9:
    #         break
    print("-------------")

    # dataloader = DataLoader(dataset, batch_size=512, shuffle=False, pin_memory=True, drop_last=True, num_workers=4)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=True, num_workers=0)
    print("--- dataloader ---")
    print(len(dataloader))
    # for i, item in enumerate(dataloader):
    # print(item)
    # if i == 9:
    #     break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with Timelapse("Looping through dataloader"):
        for X_batch, y_batch in tqdm(dataloader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pass

    print("-------------")
    # debug everything, including iterating and lenghts of everything and dataloaders
    # also check speed of dataloaders
