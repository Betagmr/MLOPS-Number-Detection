from pathlib import Path

import pytorch_lightning as plt
import torch


def save_model(model: plt.LightningModule) -> None:
    model_path = Path() / "model.pt"
    script = model.to_torchscript()

    torch.jit.save(script, model_path)
