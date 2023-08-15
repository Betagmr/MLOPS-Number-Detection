from typing import Any

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.models.dataset import CustomDataset


def evaluate_model(
    x_test: np.ndarray[Any, Any],
    y_test: np.ndarray[Any, Any],
    model: pl.LightningModule,
) -> None:
    tensor_board = TensorBoardLogger("lightning_logs", name="my_model")
    dataset = CustomDataset(x_test, y_test)
    trainer = pl.Trainer(max_epochs=2, accelerator="cuda", logger=tensor_board)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)

    trainer.test(model, dataloader)
