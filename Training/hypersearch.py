import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, Tuple, Any

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn
# import torch.optim as optim  # If your model handles optimizer internally, you don't need this.

from Training.range_doppler_dataset import RangeDopplerDataset
from model_trainer import ModelTrainer
import Training.training_utils as t_utils

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

##------------------------------------------------ MODELS --##
from NNs.CNN_Paper import Net as CNNModel
MODELS_TO_TRAIN = [CNNModel]

##------------------------------------------- DATA SOURCE --##
TRAIN_PART = 0.8
VAL_PART = 0.1
TEST_PART = 0.1

##---------------------------- CONFIG / HPARAMS / METRICS --##
DEVICE = "cuda"
RUN_CODE = "A1"
LOGS_PATH = "./Training/logs"
CONFIG = {
    "batch_size": 32,
    "early_stopping_patience": 10,
    "early_stopping_threshold": 0.01,
}

HPARAM_BOUNDS = {
    "learning_rate": ([1e-5, 1e-2], t_utils.HPARAM_SWEEP_LOG),
    "dropout_rate": ([0.2, 0.7], t_utils.HPARAM_SWEEP_LIN),
}

# METRICS IN model_trainer.py

## ------------------------------------------ DATALOADERS --##
train_dataset = RangeDopplerDataset(range_start=0, range_end=TRAIN_PART)
val_dataset = RangeDopplerDataset(range_start=TRAIN_PART, range_end=TRAIN_PART + VAL_PART)
test_dataset = RangeDopplerDataset(range_start=TRAIN_PART + VAL_PART, range_end=1)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)


# ------------------------------------------------- SWEEP --##
def hparams_sweep(num_trials: int = 8):

    for model_class in MODELS_TO_TRAIN:
        print(f"Starting sweep random search for model: {model_class.__name__}")

        for trial in range(1, num_trials + 1):

            hparams = t_utils.sample_hparams(HPARAM_BOUNDS)

            model = model_class(
                learning_rate=hparams.get("learning_rate"),
                dropout_rate=hparams.get("dropout_rate"),
            )

            optimizer = optim.Adam(model.parameters(), lr=hparams.get("learning_rate"))

            run_time = time.strftime("%d-%m_%H-%M")
            run_name = f"{run_time}_{model_class.__name__}_{RUN_CODE}"

            trainer = ModelTrainer(
                device=DEVICE,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
                hparams=hparams,
                config=CONFIG,
                run_name=run_name,
                log_dir=LOGS_PATH
            )

            print(f"[{model_class.__name__}] Finished trial {trial}/{num_trials}")

            trainer.train()


if __name__ == "__main__":
    hparams_sweep()