from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot


def process_data(data_path: Path) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    raw: pd.DataFrame = pd.read_csv(data_path)

    num_images = raw.shape[0]
    x_as_array = raw.values[:, 1:]
    x_shaped_array = x_as_array.reshape(num_images, 1, 28, 28)

    out_x = x_shaped_array / 255
    out_y = one_hot(torch.tensor(raw.label), num_classes=10).numpy()

    return out_x, out_y


if __name__ == "__main__":
    out_x, out_y = process_data(data_path=Path("data/train.csv"))

    print("Number of elements in test", len(out_y))
