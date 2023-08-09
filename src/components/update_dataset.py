from pathlib import Path

import pandas as pd
from clearml import Dataset


def update_dataset(dataset_name: str) -> None:
    dataset_manager = Dataset.get(dataset_name=dataset_name, alias=dataset_name)

    data_path = Path(dataset_manager.get_local_copy()) / "train.csv"
    train_df = pd.read_csv(data_path)

    labels = [str(i) for i in range(10)]
    values = [[element] for element in train_df.label.value_counts().sort_index().values]

    dataset_manager.get_logger().report_histogram(
        title="Label distribution",
        series="Label distribution",
        labels=labels,
        values=values,
    )


if __name__ == "__main__":
    update_dataset(dataset_name="digit_dataset")
