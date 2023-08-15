from clearml import Dataset


def download_dataset(dataset_name: str, mut_copy: bool = False) -> tuple[str, str]:
    dataset_manager = Dataset.get(dataset_name=dataset_name, alias=dataset_name)

    if mut_copy:
        return (
            dataset_manager.get_mutable_local_copy(
                target_folder="./data",
                overwrite=True,
            ),
            dataset_manager.id,
        )

    return dataset_manager.get_local_copy(), dataset_manager.id


if __name__ == "__main__":
    url, _ = download_dataset(
        dataset_name="digit_dataset",
        mut_copy=True,
    )
    print("Data stored in: ", url)
