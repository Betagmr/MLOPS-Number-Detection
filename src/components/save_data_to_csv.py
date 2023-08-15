from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.types import np_array


def save_data_to_csv(input_data: np_array, labels_encoded: np_array, filepath: str | Path) -> None:
    n_pixels = 28 * 28
    n_elements = input_data.shape[0]
    columns = [f"pixel{i}" for i in range(n_pixels)]

    data = input_data.reshape(n_elements, n_pixels)
    labels = np.argmax(labels_encoded, axis=1)

    df = pd.DataFrame(data, columns=columns)
    df.insert(0, "label", labels)

    df.to_csv(filepath, index=False)
