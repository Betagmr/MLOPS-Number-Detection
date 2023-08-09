import os
from pathlib import Path

import pytorch_lightning as plt
import torch
from clearml import OutputModel, Task


def save_model(model: plt.LightningModule, task: Task = None) -> None:
    model_path = Path() / "model.pt"
    script = model.to_torchscript()

    torch.jit.save(script, model_path)

    if task:
        output_model = OutputModel(task=task, tags=["lightning"])
        output_model.update_weights(model_path)
