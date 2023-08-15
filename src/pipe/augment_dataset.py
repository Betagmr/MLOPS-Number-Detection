from clearml import Dataset, PipelineDecorator

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
    import pandas as pd

    n_pixels = 28 * 28

    data = np.concatenate(x_data, axis=0)
    data = data.reshape(data.shape[0], n_pixels)

    labels = np.concatenate(y_data, axis=0)
    labels = np.argmax(labels, axis=1)

    df = pd.DataFrame(data, columns=[f"pixel{i}" for i in range(n_pixels)])
    df.insert(0, "label", labels)

    filepath = f"{path}/train_augmented.csv"
    df.to_csv(filepath, index=False)

    return filepath


@PipelineDecorator.component(return_values=[])
def s5_create_new_dataset(dataset_name: str, resource_path: str) -> None:
    from clearml import Dataset

    from src.settings import metadata

    old_dataset = Dataset.get(
        dataset_name=dataset_name,
        alias=dataset_name,
        dataset_project=metadata.PROJECT_NAME,
    )

    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=metadata.PROJECT_NAME,
        dataset_tags=["augmented"],
        parent_datasets=[old_dataset],
    )

    dataset.add_files(resource_path)
    dataset.finalize(auto_upload=True)


@PipelineDecorator.pipeline(
    name="Augment dataset",
    project=metadata.PROJECT_NAME,
    version="0.0.1",
    args_map={"General": ["dataset_id"]},
)
def run_pipeline(dataset_id: str) -> None:
    data_path, _ = s1_download_dataset(dataset_id=dataset_id)
    x_data, y_data = s2_process_data(data_path=data_path)
    x_augmented, y_augmented = s3_augment_data(x_data=x_data, y_data=y_data, n_samples=2)
    filepath = s4_save_to_csv(x_data=x_augmented, y_data=y_augmented, path=data_path)
    s5_create_new_dataset(dataset_name=dataset_id, resource_path=filepath)

    print(f"Augmented dataset saved to {filepath}")


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    run_pipeline(dataset_id="digit_dataset")
