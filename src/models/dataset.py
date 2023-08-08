from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class CustomDataset(Dataset[Any]):
    def __init__(
        self,
        x_data: np.ndarray[Any, Any],
        y_data: np.ndarray[Any, Any],
    ) -> None:
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y
