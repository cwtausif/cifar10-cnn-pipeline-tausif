from pathlib import Path
import csv
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self, num_filters, dropout_rate):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * 2 * 16 * 16, 128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    config = params["train"]

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    epochs = config["num_epochs"]
    num_filters = config["num_filters"]
    dropout_rate = config["dropout_rate"]

    processed_dir = Path("data/processed")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    train_data = torch.load(processed_dir / "train.pt", weights_only=False)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    model = CNN(num_filters, dropout_rate)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy
        })

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss: {epoch_loss:.4f} "
            f"- accuracy: {epoch_accuracy:.4f}"
        )

    torch.save(model.state_dict(), models_dir / "model.pth")

    with open(models_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "loss", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(history)

    print("Model saved to models/model.pth")
    print("Training history saved to models/history.csv")


if __name__ == "__main__":
    main()