from pathlib import Path

import pytest

from src.components.download_dataset import download_dataset


@pytest.mark.parametrize(
    "dataset_path, mut_copy",
    [
        (
            Path().cwd() / "data",
            True,
        ),
        (
            Path().home() / ".clearml/cache/storage_manager/datasets/ds_7ff4ba86501f405baaab3bcaef015c02",
            False,
        ),
    ],
)
def test_download_dataset(dataset_path: Path, mut_copy: bool) -> None:
    url, dataset_id = download_dataset(
        dataset_name="digit_dataset",
        mut_copy=mut_copy,
    )

    assert url == str(dataset_path)
    assert dataset_id is not None
