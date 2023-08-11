import os
from pathlib import Path

from clearml import PipelineDecorator

from src.settings import metadata
from src.utils.types import np_array


@PipelineDecorator.component(return_values=["data_path"])
def s1_download_dataset(dataset_id: str) -> str:
    from src.components.download_dataset import download_dataset

    return download_dataset(dataset_id)


@PipelineDecorator.component(return_values=["x_data", "y_data"])
def s2_process_data(data_path: str) -> tuple[np_array, np_array]:
    from src.components.process_data import process_data

    return process_data(f"{data_path}/train.csv")


@PipelineDecorator.component(return_values=["x_augmented", "y_augmented"])
def s3_augment_data(x_data: np_array, y_data: np_array) -> tuple[np_array, np_array]:
    from src.components.data_augmentation import data_augmentation

    return data_augmentation(x_data, y_data)


@PipelineDecorator.pipeline(
    name="Augment dataset",
    project=metadata.PROJECT_NAME,
    version="0.0.1",
    args_map={"General": ["dataset_id"]},
)
def run_pipeline(dataset_id: str) -> None:
    data_path = s1_download_dataset(dataset_id)
    x_data, y_data = s2_process_data(data_path)
    _, _ = s3_augment_data(x_data, y_data)


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    run_pipeline(dataset_id="digit_dataset")
