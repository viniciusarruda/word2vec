import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim, mode="sum")
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear1(x)
        return x


def get_model(name: str, vocab_size: int, embedding_dim: int):
    if name == "CBOW":
        return CBOW(vocab_size, embedding_dim)
