from pathlib import Path

from clearml import Task

from src.components.download_dataset import download_dataset
from src.components.evaluate_model import evaluate_model
from src.components.process_data import process_data
from src.components.save_model import save_model
from src.components.train_model import train_model
from src.settings import metadata
from src.settings.params import TRAINING_PARAMS
from src.utils.logger import logger
from src.utils.set_seed import set_seed


def start_training(create_task: bool = False) -> None:
    logger.info("Starting training")
    if create_task:
        task: Task = Task.init(
            project_name=metadata.PROJECT_NAME,
            task_name="Model training",
            output_uri=True,
        )

        task.connect(TRAINING_PARAMS, "training_params")

    set_seed(TRAINING_PARAMS["seed"])

    logger.info("Downloading data")
    data_path = download_dataset("digit_dataset")

    logger.info("Processing data")
    train_path = Path(data_path) / "train.csv"
    x_train, y_train = process_data(train_path)

    logger.info("Training model")
    model = train_model(x_train, y_train, TRAINING_PARAMS)
    save_model(model)

    logger.info("Evaluating model")
    evaluate_model(x_train, y_train, model)


if __name__ == "__main__":
    start_training(True)
