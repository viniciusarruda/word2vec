from tqdm import tqdm
import os
from data import Vocabulary
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self, model, device, n_epochs: int, vocab: Vocabulary, train_dataloader, val_dataloader, save_folder_path: str
    ) -> None:
        self.model = model
        self.device = device
        self.n_epochs = n_epochs
        self.vocab = vocab
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.025)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: (self.n_epochs - epoch) / self.n_epochs,
        )

        self.save_folder_path = save_folder_path
        os.makedirs(os.path.join(self.save_folder_path, "tensorboard"))
        self.writer = SummaryWriter(os.path.join(self.save_folder_path, "tensorboard"))

    def train(self):
        self.model.to(self.device)

        X_batch, _ = next(iter(self.train_dataloader))
        self.writer.add_graph(self.model, X_batch[0].unsqueeze(0).to(self.device))

        for name, weight in self.model.named_parameters():
            self.writer.add_histogram(name.replace(".", "/"), weight, 0)

        for epoch in tqdm(range(1, self.n_epochs + 1), desc="Training epochs"):
            self._train_epoch(epoch)
            if self.val_dataloader is not None:
                self._validate_epoch(epoch)
        self._save_model()
        # TODO
        # ja que coloquei coisa do tensorboard aqui, colocar as coisas do tensorboard dentro do save_model aqui
        self.writer.close()

    def _train_epoch(self, epoch: int):
        mean_loss = 0
        self.model.train()

        self.writer.add_scalar("train/lr", self.lr_scheduler.get_last_lr()[0], epoch)

        for i, (X_batch, y_batch) in enumerate(
            tqdm(self.train_dataloader, desc=f"Training epoch #{epoch}", leave=False),
            start=(epoch - 1) * len(self.train_dataloader) + 1,
        ):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            y_pred = self.model(X_batch)
            loss = self.loss_fn(y_pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_loss += loss.item()
            self.writer.add_scalar("train/loss/mean_batch", loss.item() / X_batch.size(0), i)

        self.lr_scheduler.step()

        self.writer.add_scalar("train/loss/mean_epoch", mean_loss / len(self.train_dataloader.dataset), epoch)

        for name, weight in self.model.named_parameters():
            self.writer.add_histogram(name.replace(".", "/"), weight, epoch)
            self.writer.add_histogram(f"{name.replace('.', '/')}.grad", weight.grad, epoch)

    def _validate_epoch(self, epoch: int):
        acc = 0
        mean_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, (X_batch, y_batch) in enumerate(
                tqdm(
                    self.val_dataloader,
                    desc=f"Validating epoch #{epoch}",
                    leave=False,
                ),
                start=(epoch - 1) * len(self.val_dataloader) + 1,
            ):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)

                mean_loss += loss.item()
                self.writer.add_scalar("val/loss/mean_batch", loss.item() / X_batch.size(0), i)

                acc += (y_pred.argmax(dim=1) == y_batch).float().sum().item()

            self.writer.add_scalar("val/loss/mean_epoch", mean_loss / len(self.val_dataloader.dataset), epoch)
            self.writer.add_scalar("val/acc_epoch", acc / len(self.val_dataloader.dataset), epoch)

    def _save_model(self) -> None:
        embeddings = self.model.embeddings.weight.detach().cpu().numpy()
        self.writer.add_embedding(embeddings, metadata=self.vocab.itos(), tag="embeddings")

        # model_folder = os.path.join(self.save_folder_path, "model")
        # os.makedirs(model_folder)
        # np.save(os.path.join(model_folder, "word_embeddings.npy"), embeddings)
        self.save_embeddings()

    def save_embeddings(self):
        # TODO
        # this is the wrong place to put this
        # it should be vocab based, considering special tokens to remove it
        embeddings = self.model.embeddings.weight.detach().cpu().numpy()

        model_folder = os.path.join(self.save_folder_path, "model")
        os.makedirs(model_folder)
        n = (
            len(self.vocab.word_to_idx) - 2
        )  # excluding special tokens, I know they are the first two ones but this is not save to do
        d = embeddings.shape[1]
        print(">>>>>>>>>>>>>>>> embeddings.shape", embeddings.shape)

        # rows = (f"\n{w} {' '.join(embeddings[i, :])}" for i, w in self.vocab.idx_to_word.items())
        # next(rows)  # skipping <pad>
        # next(rows)  # skipping <unk>

        with open(os.path.join(model_folder, "word_embeddings.txt"), "w", encoding="utf-8") as f:
            f.write(f"{n} {d}")
            # f.writelines(rows)
            for i, w in self.vocab.idx_to_word.items():
                if i > 1:  # skipping <pad> and <unk>
                    f.write(f"\n{w} ")
                    np.savetxt(f, embeddings[i, np.newaxis], fmt="%.12f", newline="")


class Evaluator:
    def __init__(self, model, device, dataloader) -> None:
        self.model = model
        self.device = device
        self.dataloader = dataloader

    def evaluate(self):
        acc = 0
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in tqdm(self.dataloader, desc="Evaluating"):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)

                acc += (y_pred.argmax(dim=1) == y_batch).float().sum().item()
        print(f"Correct: {acc} from {len(self.dataloader.dataset)}")
        acc /= len(self.dataloader.dataset)
        print("len test set: ", len(self.dataloader.dataset))
        print("acc: ", acc)
