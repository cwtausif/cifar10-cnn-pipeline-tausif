import yaml
import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    def __init__(self, num_filters, dropout_rate):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((num_filters * 2) * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def main():
    params = load_params()
    train_params = params["train"]

    model = CIFAR10CNN(
        num_filters=train_params["num_filters"],
        dropout_rate=train_params["dropout_rate"],
    )

    print("CIFAR-10 CNN model created")
    print(model)


if __name__ == "__main__":
    main()