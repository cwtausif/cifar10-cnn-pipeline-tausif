from pathlib import Path
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from train import CNN


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    config = params["train"]

    processed_dir = Path("data/processed")
    models_dir = Path("models")

    test_data = torch.load(
    processed_dir / "test.pt",
    weights_only=False
)

    test_loader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=False
    )

    model = CNN(
        config["num_filters"],
        config["dropout_rate"]
    )

    model.load_state_dict(
        torch.load(models_dir / "model.pth")
    )

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.numpy())

    test_loss = total_loss / total
    test_accuracy = correct / total

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix
    )

    display.plot()
    plt.title("CIFAR-10 Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    print("Metrics saved to metrics.json")
    print("Confusion matrix saved to models/confusion_matrix.png")


if __name__ == "__main__":
    main()