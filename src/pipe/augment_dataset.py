from clearml import PipelineDecorator

from src.settings import metadata
from src.utils.types import np_array


@PipelineDecorator.component(return_values=["data_path", "parent_id"])
def s1_download_dataset(dataset_id: str) -> tuple[str, str]:
    from src.components.download_dataset import download_dataset

    return download_dataset(dataset_id)


@PipelineDecorator.component(return_values=["x_data", "y_data"])
def s2_process_data(data_path: str) -> tuple[np_array, np_array]:
    from src.components.process_data import process_data

    return process_data(f"{data_path}/train.csv")


@PipelineDecorator.component(return_values=["x_augmented", "y_augmented"])
def s3_augment_data(x_data: np_array, y_data: np_array, n_samples: int) -> tuple[list[np_array], list[np_array]]:
    from src.components.data_augmentation import data_augmentation

    data, labels = [], []
    for _ in range(n_samples):
        x_aug, y_aug = data_augmentation(x_data, y_data)
        data.append(x_aug)
        labels.append(y_aug)

    return data, labels


@PipelineDecorator.component(return_values=["filepath"])
def s4_save_to_csv(x_data: list[np_array], y_data: list[np_array], path: str) -> str:
    import numpy as np

    from src.components.save_data_to_csv import save_data_to_csv

    data = np.concatenate(x_data, axis=0)
    labels = np.concatenate(y_data, axis=0)
    filepath = f"{path}/train_augmented.csv"

    save_data_to_csv(data, labels, filepath)

    return filepath


@PipelineDecorator.component(return_values=["new_dataset_id"])
def s5_create_new_dataset(dataset_name: str, resource_path: str, parent_id: str) -> str:
    from clearml import Dataset

    from src.settings import metadata

    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=metadata.PROJECT_NAME,
        dataset_tags=["augmented"],
        parent_datasets=[parent_id],
    )

    dataset.add_files(resource_path)
    dataset.finalize(auto_upload=True)

    return dataset.id


@PipelineDecorator.pipeline(
    name="Augment dataset",
    project=metadata.PROJECT_NAME,
    version="0.0.1",
    args_map={"General": ["dataset_id", "n_augmentations"]},
)
def run_pipeline(dataset_id: str, n_augmentations: int) -> None:
    data_path, parent_id = s1_download_dataset(dataset_id)
    x_data, y_data = s2_process_data(data_path)
    x_augmented, y_augmented = s3_augment_data(x_data, y_data, n_augmentations)
    filepath = s4_save_to_csv(x_augmented, y_augmented, data_path)
    new_dataset_id = s5_create_new_dataset(dataset_id, filepath, parent_id)

    print(f"Augmented dataset saved to {filepath}")


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    run_pipeline(dataset_id="digit_dataset", n_augmentations=3)
