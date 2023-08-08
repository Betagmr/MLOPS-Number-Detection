from typing import TypedDict


class TrainingParams(TypedDict):
    lr: float
    batch_size: int
    n_epochs: int
    seed: int
    device: str
    is_checkpoint: bool


TRAINING_PARAMS: TrainingParams = {
    "lr": 1e-3,
    "batch_size": 32,
    "n_epochs": 2,
    "seed": 42,
    "device": "cuda",
    "is_checkpoint": False,
}
