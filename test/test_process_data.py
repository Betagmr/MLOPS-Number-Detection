from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.components.process_data import process_data
from src.components.save_data_to_csv import save_data_to_csv


@pytest.fixture()
def data_path(tmp_path: Path) -> Path:
    file_path = tmp_path / "data.csv"

    return file_path


def test_save_data(data_path: str) -> None:
    input_data = np.array(
        [
            [0 for _ in range(28 * 28)],
            [1 for _ in range(28 * 28)],
            [0 for _ in range(28 * 28)],
        ]
    )

    labels = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    )

    save_data_to_csv(input_data, labels, data_path)
    df = pd.read_csv(data_path)

    assert df["label"].tolist() == [0, 3, 7]
    assert df["pixel0"].tolist() == [0, 255, 0]


def test_load_data(data_path: Path) -> None:
    # df = pd.read_csv(data_path)
    # print(df.head())

    assert 1 == 1
