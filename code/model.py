import torch.nn as nn
from prettytable import PrettyTable


class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean", max_norm=1, padding_idx=0)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear1(x)
        return x


def print_model_parameters(model) -> None:
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if len(table._dividers) > 0:
        table._dividers[-1] = True

    table.add_row(["Total", total_params])

    print(table)


def get_model(name: str, vocab_size: int, embedding_dim: int) -> nn.Module:
    if name == "CBOW":
        model = CBOW(vocab_size, embedding_dim)
    else:
        raise ValueError(f"Model {name} is not implemented!")

    print(f"Loaded model {name}")
    print_model_parameters(model)

    return model
