from typing import Any

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.models.dataset import CustomDataset
from src.models.torch_model import LitModel
from src.settings.params import TrainingParams


def train_model(
    x_train: np.ndarray[Any, Any],
    y_train: np.ndarray[Any, Any],
    params: TrainingParams,
) -> pl.LightningModule:
    # Set up the trainer
    tensor_board = TensorBoardLogger(
        save_dir="lightning_logs",
        name="my_model",
    )
    trainer = pl.Trainer(
        max_epochs=params["n_epochs"],
        accelerator=params["device"],
        logger=tensor_board,
        enable_checkpointing=params["is_checkpoint"],
    )

    # Train the model
    model = LitModel(lr=params["lr"])
    dataset = CustomDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    trainer.fit(model, dataloader)

    return model
