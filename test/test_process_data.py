from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.components.process_data import process_data
from src.components.save_data_to_csv import save_data_to_csv
from src.utils.types import np_array


@pytest.fixture(scope="function")
def data_path(tmp_path: Path) -> Path:
    file_path = tmp_path / "data.csv"

    return file_path


@pytest.fixture()
def sample_data() -> tuple[np_array, np_array]:
    input_data = np.array(
        [
            [0 for _ in range(28 * 28)],
            [1 for _ in range(28 * 28)],
            [15 / 255 for _ in range(28 * 28)],
        ]
    )

    labels = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    )

    return input_data, labels


def test_save_data(sample_data: tuple[np_array, np_array], data_path: str) -> None:
    input_data, labels = sample_data
    save_data_to_csv(input_data, labels, data_path)

    df = pd.read_csv(data_path)

    assert df["label"].tolist() == [0, 3, 7]
    assert df["pixel0"].tolist() == [0, 255, 15]


def test_load_data(sample_data: tuple[np_array, np_array], data_path: Path) -> None:
    input_data, labels = sample_data
    save_data_to_csv(input_data, labels, data_path)

    x_data, y_data = process_data(data_path)

    assert x_data[:, 0, 0, 0].tolist() == [0, 1, 15 / 255]

    assert y_data.tolist()[0] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert y_data.tolist()[1] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert y_data.tolist()[2] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
