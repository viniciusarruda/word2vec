import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import time


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


def _load_processed_file_data(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    data = data.replace("\n", "<eos>")
    data = data.split()

    assert all([len(d) > 0 for d in data])

    return data


class Vocabulary:
    def __init__(
        self, data: list[str] = None, folder_path: str = None, min_freq: int = 1, default_token: str = "<unk>"
    ) -> None:
        if data is not None:
            vocab = self.build_vocab(data, min_freq)
            assert type(vocab) is set

            if default_token in vocab:
                print(f"WARNING: Default index ({default_token}) found in data")
            else:
                vocab.add(default_token)

            self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
            self.idx_to_word = {idx: word for idx, word in enumerate(vocab)}

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

    def stoi(self, data: list[str]) -> list[int]:
        return [self.word_to_idx.get(word, self.default_idx) for word in data]

    def itos(self, idxs: list[int] | None = None) -> list[str]:
        if idxs is None:
            idxs = range(len(self))
        return [self.idx_to_word[idx] for idx in idxs]

    def save_data_stats(self, data: list[str], filename: str, folder_path: str) -> None:
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
    def build_vocab(data: list[str], min_freq: int) -> set:
        data_freq = Counter(data)
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
    def __init__(self, data_as_idxs: list[int], context_size: int):
        self.context_size = context_size
        self.data_as_idxs = data_as_idxs

    def __len__(self):
        return len(self.data_as_idxs) - 2 * self.context_size

    def __getitem__(self, idx):
        idx += self.context_size
        left_context = self.data_as_idxs[idx - self.context_size : idx]
        center_idx = self.data_as_idxs[idx]
        right_context = self.data_as_idxs[idx + 1 : idx + self.context_size + 1]
        context_idxs = left_context + right_context

        context_input = torch.tensor(context_idxs, dtype=torch.long)

        return context_input, center_idx


def get_processed_data(context_size: int, dataset_config: dict, dataloader_config: dict, output_folder_path: str) -> dict:
    # TODO colocar docstring automaticamente

    dataset_names = dataloader_config.keys()

    stats_folder_path = os.path.join(output_folder_path, "stats")
    os.makedirs(stats_folder_path)

    vocab_folder_path = os.path.join(output_folder_path, "vocab")

    datasets = {}
    if "train" in dataset_names:
        os.makedirs(vocab_folder_path)
        train_data = _load_processed_file_data(dataset_config["train"]["file_path"])

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
            data = _load_processed_file_data(dataset_config[name]["file_path"])

            vocab.save_data_stats(data, name, stats_folder_path)
            data_as_idxs = vocab.stoi(data)
            datasets[name] = ContextDataset(data_as_idxs, context_size)

    dataloaders = {}
    for name in dataset_names:
        print(name, dataloader_config[name])
        dataloaders[name] = DataLoader(datasets[name], **dataloader_config[name])

    return dataloaders, vocab
