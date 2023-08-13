from typing import Any

import numpy as np
import torch
from torchvision import transforms as T


def data_augmentation(
    x_data: np.ndarray[Any, Any],
    y_data: np.ndarray[Any, Any],
    translate: tuple[float, float] = (0.2, 0.25),
    scale: tuple[float, float] = (0.55, 0.77),
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    augmentation = T.Compose(
        [
            T.RandomRotation(10),
            T.RandomAffine(0, translate=translate, scale=scale),
        ]
    )

    x_augmented = [augmentation(element) for element in torch.from_numpy(x_data)]

    return np.array(x_augmented), y_data
