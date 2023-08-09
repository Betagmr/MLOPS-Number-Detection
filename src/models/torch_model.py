import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchmetrics
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor, nn, optim
from torch.nn import functional as F


class LitModel(pl.LightningModule):
    def __init__(self, lr: float = 0.0001) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=40 * 14 * 14, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1),
        )

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=10)

        # Outputs
        self.test_outputs: list[Tensor] = []
        self.test_targets: list[Tensor] = []

        # Hyperparameter
        self.lr = lr

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_predicted = self.forward(x)
        metrics = self.get_metics(y_predicted, y)
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)

        return metrics["loss"]

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        y_predicted = self.forward(x)

        if batch_idx % 50 == 0 and self.logger is not None:
            self.logger.experiment.add_images(
                f"{y_predicted.argmax(dim=1).tolist()[:5]}",
                x[:5],
            )

        metrics = self.get_metics(y_predicted, y)
        self.log_dict(metrics, prog_bar=True, on_step=False)
        self.test_outputs.append(y_predicted.argmax(dim=1).cpu())
        self.test_targets.append(y.argmax(dim=1).cpu())

    def on_test_epoch_end(self) -> None:
        predictions = torch.cat(list(self.test_outputs))
        targets = torch.cat(list(self.test_targets))

        ConfusionMatrixDisplay.from_predictions(targets, predictions)
        plt.show()

    def get_metics(self, y_predicted: Tensor, y: Tensor) -> dict[str, Tensor]:
        loss = F.cross_entropy(y_predicted, y)
        acc = self.accuracy(y_predicted.argmax(dim=1), y.argmax(dim=1))
        f1 = self.f1_score(y_predicted.argmax(dim=1), y.argmax(dim=1))

        return {"loss": loss, "acc": acc, "f1": f1}
