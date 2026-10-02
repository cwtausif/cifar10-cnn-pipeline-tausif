import json

import torch
import torch.nn as nn


def evaluate(model, test_loader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = total_loss / total
    accuracy = correct / total

    metrics = {
        "test_loss": loss,
        "test_accuracy": accuracy,
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    return metrics


if __name__ == "__main__":
    print("CIFAR-10 model evaluation")