import os
import json
import numpy as np
import torch
import toml
from datetime import datetime
from data import get_processed_data
from model import get_model
from executor import Trainer, Evaluator
from visualization import plot_embeddings, plot_embeddings_from_file
from prettytable import PrettyTable
import random


def load_config(config_file_path: str = "config.toml") -> dict:
    config = toml.load(config_file_path)

    if "save_path" not in config["experiment"]:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        config["experiment"]["save_path"] = os.path.join(
            os.path.normpath(config["experiment"]["folder_path"]), config["dataset"]["name"], config["model"]["name"], timestamp
        )

    os.makedirs(config["experiment"]["save_path"])

    config["dataset"] = config["dataset"][config["dataset"]["name"]]

    # It will save the parsed config file
    with open(os.path.join(config["experiment"]["save_path"], "config.toml"), "w", encoding="utf-8") as f:
        toml.dump(config, f)

    return config


def main():
    print("-- 0 --")
    config = load_config()

    # determinism
    random.seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])
    torch.manual_seed(config["experiment"]["seed"])
    torch.cuda.manual_seed_all(config["experiment"]["seed"])
    os.environ["PYTHONHASHSEED"] = str(
        config["experiment"]["seed"]
    )  # I think I need to set this before lauching python interpreter

    # TODO this slows down training, so put a flag here and set it at config file
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    print("-- 1 --")
    dataloaders, vocab = get_processed_data(
        config["model"]["context_size"],
        config["dataset"],
        config["dataloader"],
        config["experiment"]["save_path"],
    )

    # for i, (X_batch, y_batch) in enumerate(tqdm(dataloaders["train"])):
    #     tqdm.write(f"-- batch #{i} --")
    #     tqdm.write(f"X_batch: {X_batch.size()}")
    #     tqdm.write(f"y_batch: {y_batch.size()}")
    #     tqdm.write("\n")
    # exit()

    model = get_model(config["model"]["name"], len(vocab), config["model"]["embedding_dim"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # add verbose option
    table = PrettyTable(["Field", "Length"])
    for k, v in dataloaders.items():
        table.add_row([f"dataloaders['{k}'].dataset", len(v.dataset)])
        table.add_row([f"dataloaders['{k}']", len(v)], divider=True)
    print(table)
    print(f"Length vocab: {len(vocab)}")

    trainer = Trainer(
        model,
        device,
        config["train"]["n_epochs"],
        vocab,
        dataloaders["train"],
        dataloaders.get("validation", None),
        config["experiment"]["save_path"],
    )
    trainer.train()

    if "test" in dataloaders:
        evaluator = Evaluator(model, device, dataloaders["test"])
        evaluator.evaluate()

    exit()

    # plot_embeddings(
    #     config["visualization"]["images_folder_path"], "all_{plot_type}.png", trainer.numpy_embeddings, vocab.word_to_idx.keys()
    # )

    plot_embeddings_from_file(
        "./data/countries_capital.json",
        config["visualization"]["images_folder_path"],
        "capitals_{plot_type}.png",
        trainer.numpy_embeddings,
        vocab,
    )

    plot_embeddings_from_file(
        "./data/google-10000-english-usa-no-swears.txt",
        config["visualization"]["images_folder_path"],
        "most_common_{plot_type}.png",
        trainer.numpy_embeddings,
        vocab,
    )

    print(len(word2Ind))
    words = [word for word in word2Ind]
    # given a list of words and the embeddings, it returns a matrix with all the embeddings
    idxs = [word2Ind[word] for word in words]
    X = word_embeddings[idxs, :].numpy()

    result = compute_pca(X, 2)
    plot_words(result, words, "pca_all.png")
    tsne(X, words, "tsne_all.png")

    with open("./data/countries_capital.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # data = data[:100]
    countries = [entry["country"].lower() for entry in data if entry["city"] is not None]
    capitals = [entry["city"].lower() for entry in data if entry["city"] is not None]
    words = countries + capitals
    words = [word for word in words if word in word2Ind]

    # given a list of words and the embeddings, it returns a matrix with all the embeddings
    idxs = [word2Ind[word] for word in words]
    X = word_embeddings[idxs, :].numpy()

    result = compute_pca(X, 2)
    plot_words(result, words, "pca_countries.png")

    tsne(X, words, "tsne_countries.png")

    norms = (word_embeddings**2).sum(axis=1, keepdim=True) ** (1 / 2)
    embeddings_norm = word_embeddings / norms

    emb1 = embeddings_norm[word2Ind["canada"]]
    emb2 = embeddings_norm[word2Ind["ottawa"]]
    emb3 = embeddings_norm[word2Ind["brazil"]]

    emb4 = emb1 - emb2 + emb3
    emb4_norm = (emb4**2).sum() ** (1 / 2)
    emb4 = emb4 / emb4_norm

    # change to numpy
    emb4 = np.reshape(emb4, (len(emb4), 1))
    dists = np.matmul(embeddings_norm, emb4).flatten()

    top5 = np.argsort(-dists)[:10]

    for word_id in top5:
        print("{}: {:.3f}".format(Ind2word[word_id.item()], dists[word_id]))


# 1. revisar
# 2. overfit small data
# 3. check embeddings visualizing them
# 4. compare with other embedding like spacy?
# 5. train with bigger data
# 6. skipgram
# 7. fasttext
# 8. DONE! -> coursera!!!


if __name__ == "__main__":
    main()
