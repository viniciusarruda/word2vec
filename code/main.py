import os
import numpy as np
import torch
import toml
from datetime import datetime
from data import get_processed_data
from model import get_model
from executor import Trainer, Evaluator
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

    dataloaders, vocab = get_processed_data(
        config["model"]["context_size"],
        config["dataset"],
        config["dataloader"],
        config["experiment"]["save_path"],
    )

    model = get_model(config["model"]["name"], len(vocab), config["model"]["embedding_dim"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


if __name__ == "__main__":
    main()
