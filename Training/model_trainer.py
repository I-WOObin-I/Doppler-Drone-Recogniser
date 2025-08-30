import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import training_utils as t_utils


class ModelTrainer:
    def __init__(self, device, model, train_loader, val_loader, optimizer, criterion, hparams, config, run_name, log_dir="./logs"):
        self.device = device
        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.criterion = criterion

        self.hparams = hparams
        self.config = config

        self.metrics = {}
        self.best_metrics = {}
        self.run_name = run_name

        self.best_val_loss = np.inf
        self.epochs_no_improve = 0
        self.writer = SummaryWriter(f"{log_dir}/{run_name}")


# ------------------------------------------------- TRAIN --##

    def train(self):

        print(f"\nStarting Training: {self.hparams}")
        
        for epoch in range(self.hparams["num_epochs"]):

            print(f"\nEpoch {epoch+1}/{self.hparams['num_epochs']}")

            train_loss, train_acc, train_time = self._train_one_epoch()

            val_loss, val_acc, val_time, confusion_matrix_fig = self._validate(epoch)

            self._log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, train_time, val_time)

            self._save_if_best(val_loss)
            if self._check_early_stopping():
                self.writer.add_figure("Confusion Matrix", confusion_matrix_fig, epoch)
                break


        for key, value in self.metrics.items():
            self.hparams["hparam/" + key] = value

        self.writer.add_hparams(hparam_dict=self.hparams, metric_dict={})

        self.writer.close()

    def _train_one_epoch(self):
        
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        duration = time.time() - start_time
        train_loss = total_loss / len(self.train_loader)
        train_acc = 100 * correct / total

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% ({duration:.2f}s)")

        return train_loss, train_acc, duration

    def _validate(self, epoch):
        
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        start_time = time.time()

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        duration = time.time() - start_time
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100 * correct / total

        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% ({duration:.2f}s)")

        confusion_matrix_fig = t_utils._log_confusion_matrix(self, all_labels, all_preds, ["Without Drone", "With Drone"], epoch)

        return val_loss, val_acc, duration, confusion_matrix_fig


# ------------------------------------------------- UTILS --##
    def _log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, train_time, val_time):
        self.metrics = {
            "loss/train": train_loss,
            "loss/val": val_loss,
            "accuracy/train": train_acc,
            "accuracy/val": val_acc,
            "duration/train": train_time,
            "duration/val": val_time,
            "epoch": epoch,
        }

        for key, value in self.metrics.items():
            self.writer.add_scalar(key, value, epoch)

        print(f"saved logs to {self.writer.log_dir}")


    def _save_if_best(self, val_loss):
        if val_loss < self.best_val_loss:
            for key, value in self.metrics.items():
                self.best_metrics["hparam/" + key] = value

            torch.save(self.model.state_dict(), f"./models/{self.run_name}.pth")
            print("Model saved (new best validation loss)")
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1


    def _check_early_stopping(self):
        if self.epochs_no_improve >= self.hparams["early stopping patience"]:
            print(f"Early stopping triggered after {self.epochs_no_improve} epochs without improvement.")
            return True
        return False
