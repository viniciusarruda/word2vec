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
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.025)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: (self.n_epochs - epoch) / self.n_epochs,
        )
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200, 400, 600, 800], gamma=0.05)

        self.save_folder_path = save_folder_path
        os.makedirs(os.path.join(self.save_folder_path, "tensorboard"))
        self.writer = SummaryWriter(os.path.join(self.save_folder_path, "tensorboard"))

    def train(self):
        self.model.to(self.device)

        X_batch, _ = next(iter(self.train_dataloader))
        self.writer.add_graph(self.model, X_batch[0].unsqueeze(0).to(self.device))

        # TODO check how much time this takes at the end
        for name, weight in self.model.named_parameters():
            self.writer.add_histogram(name, weight, 0)  # change epoch to start on 1 instead of 0, so here could be 0

        for epoch in tqdm(range(self.n_epochs), desc="Training epochs"):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
        self._save_model()
        # TODO
        # ja que coloquei coisa do tensorboard aqui, colocar as coisas do tensorboard dentro do save_model aqui
        self.writer.close()

    def _train_epoch(self, epoch: int):
        mean_loss = 0
        self.model.train()

        self.writer.add_scalar("train/lr", self.lr_scheduler.get_last_lr()[0], epoch)

        for i, (X_batch, y_batch) in enumerate(tqdm(self.train_dataloader, desc=f"Training epoch #{epoch}", leave=False)):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            y_pred = self.model(X_batch)
            loss = self.loss_fn(y_pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_loss += loss.item()
            self.writer.add_scalar(
                "train/loss/iter", loss.item(), epoch * len(self.train_dataloader) + i
            )  # TODO this loss is for the batch! I should divide it for the batch size! notie that batch could change, so use the tensor size
            # TODO should be called mean_batch
            # TODO add a new scalar to compare, if good, remove this above

        self.lr_scheduler.step()

        self.writer.add_scalar(
            "train/loss/mean", mean_loss / len(self.train_dataloader.dataset), epoch
        )  # TODO should be called mean_epoch

        # TODO check how much time this takes at the end
        for name, weight in self.model.named_parameters():
            self.writer.add_histogram(name, weight, epoch + 1)
            self.writer.add_histogram(f"{name}.grad", weight.grad, epoch + 1)  # keep ?

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
                )
            ):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)

                mean_loss += loss.item()
                self.writer.add_scalar("val/loss/iter", loss.item(), epoch * len(self.val_dataloader) + i)

                # Verbose
                # for i in range(X_batch.size(0)):
                #     context_idxs = X_batch[i].tolist()
                #     print(f"Context: {[Ind2word[idx] for idx in context_idxs]} | Pred: {Ind2word[y_pred[i].argmax().item()]} | GT: {Ind2word[y_batch[i].item()]}")

                acc += (y_pred.argmax(dim=1) == y_batch).float().sum().item()

        self.writer.add_scalar("val/loss/mean", mean_loss / len(self.val_dataloader.dataset), epoch)
        self.writer.add_scalar("val/acc", acc / len(self.val_dataloader.dataset), epoch)

    def _save_model(self) -> None:
        embeddings = self.model.embeddings.weight.detach().cpu().numpy()
        self.writer.add_embedding(embeddings, metadata=self.vocab.itos(), tag="embeddings")

        model_foder = os.path.join(self.save_folder_path, "model")
        os.makedirs(model_foder)
        np.save(os.path.join(model_foder, "word_embeddings.npy"), embeddings)


class Evaluator:
    def __init__(self, model, device, dataloader) -> None:
        self.model = model
        self.device = device
        self.dataloader = dataloader

    def evaluate(self):
        acc = 0
        # self.model.to(self.device) # already in device?
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in tqdm(self.dataloader, desc="Evaluating"):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)

                # Verbose
                # for i in range(X_batch.size(0)):
                #     context_idxs = X_batch[i].tolist()
                #     print(f"Context: {[Ind2word[idx] for idx in context_idxs]} | Pred: {Ind2word[y_pred[i].argmax().item()]} | GT: {Ind2word[y_batch[i].item()]}")

                acc += (y_pred.argmax(dim=1) == y_batch).float().sum().item()
        print(f"Correct: {acc} from {len(self.dataloader.dataset)}")
        acc /= len(self.dataloader.dataset)
        print("len test set: ", len(self.dataloader.dataset))
        print("acc: ", acc)
