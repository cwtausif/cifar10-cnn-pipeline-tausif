from pathlib import Path
import csv
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CNN(nn.Module):

    def __init__(self, num_filters, dropout_rate):

        super().__init__()

        # -----------------------------------------------------
        # BLOCK 1
        # 32 x 32 -> 16 x 16
        # -----------------------------------------------------
        self.block1 = nn.Sequential(
            nn.Conv2d(
                3,
                num_filters,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                num_filters,
                num_filters,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )

        # -----------------------------------------------------
        # BLOCK 2
        # 16 x 16 -> 8 x 8
        # -----------------------------------------------------
        self.block2 = nn.Sequential(
            nn.Conv2d(
                num_filters,
                num_filters * 2,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                num_filters * 2,
                num_filters * 2,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )

        # -----------------------------------------------------
        # BLOCK 3
        # 8 x 8 -> 4 x 4
        # -----------------------------------------------------
        self.block3 = nn.Sequential(
            nn.Conv2d(
                num_filters * 2,
                num_filters * 4,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                num_filters * 4,
                num_filters * 4,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )

        # -----------------------------------------------------
        # BLOCK 4
        # 4 x 4
        # -----------------------------------------------------
        self.block4 = nn.Sequential(
            nn.Conv2d(
                num_filters * 4,
                num_filters * 4,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(inplace=True)
        )

        # -----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # -----------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # -----------------------------------------------------
        # CLASSIFIER
        # -----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                num_filters * 4,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout_rate),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)

        return self.classifier(x)


def main():

    # ---------------------------------------------------------
    # LOAD PARAMETERS
    # ---------------------------------------------------------

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    config = params["train"]

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    epochs = config["num_epochs"]
    num_filters = config["num_filters"]
    dropout_rate = config["dropout_rate"]

    # ---------------------------------------------------------
    # DIRECTORIES
    # ---------------------------------------------------------

    processed_dir = Path("data/processed")
    models_dir = Path("models")

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # LOAD TRAINING DATA
    # ---------------------------------------------------------

    train_data = torch.load(
        processed_dir / "train.pt",
        weights_only=False
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # ---------------------------------------------------------
    # CREATE MODEL
    # ---------------------------------------------------------

    model = CNN(
        num_filters=num_filters,
        dropout_rate=dropout_rate
    )

    # ---------------------------------------------------------
    # LOSS FUNCTION
    # ---------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    # ---------------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4
    )

    # ---------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # ---------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    # ---------------------------------------------------------
    # TRAINING HISTORY
    # ---------------------------------------------------------

    history = []

    # ---------------------------------------------------------
    # TARGET ACCURACY
    # ---------------------------------------------------------

    TARGET_ACCURACY = 0.90

    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # -----------------------------------------------------
        # BATCH LOOP
        # -----------------------------------------------------

        for images, labels in train_loader:

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            # Update weights
            optimizer.step()

            # -------------------------------------------------
            # CALCULATE LOSS
            # -------------------------------------------------

            running_loss += (
                loss.item() * images.size(0)
            )

            # -------------------------------------------------
            # CALCULATE ACCURACY
            # -------------------------------------------------

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        # -----------------------------------------------------
        # UPDATE LEARNING RATE
        # -----------------------------------------------------

        scheduler.step()

        # -----------------------------------------------------
        # EPOCH METRICS
        # -----------------------------------------------------

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        current_lr = optimizer.param_groups[0]["lr"]

        # -----------------------------------------------------
        # SAVE HISTORY
        # -----------------------------------------------------

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "learning_rate": current_lr
        })

        # -----------------------------------------------------
        # PRINT RESULTS
        # -----------------------------------------------------

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss: {epoch_loss:.4f} "
            f"- accuracy: {epoch_accuracy:.4f} "
            f"- lr: {current_lr:.6f}"
        )

        # -----------------------------------------------------
        # STOP IF 90% ACCURACY IS REACHED
        # -----------------------------------------------------

        if epoch_accuracy >= TARGET_ACCURACY:

            print()
            print("=" * 60)
            print(
                f"TARGET ACCURACY REACHED: "
                f"{epoch_accuracy * 100:.2f}%"
            )
            print(
                f"Stopping training at epoch "
                f"{epoch + 1}/{epochs}"
            )
            print("=" * 60)

            # Save model immediately
            torch.save(
                model.state_dict(),
                models_dir / "model.pth"
            )

            break

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    torch.save(
        model.state_dict(),
        models_dir / "model.pth"
    )

    print()
    print("Model saved to models/model.pth")

    # ---------------------------------------------------------
    # SAVE TRAINING HISTORY
    # ---------------------------------------------------------

    with open(
        models_dir / "history.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "loss",
                "accuracy",
                "learning_rate"
            ]
        )

        writer.writeheader()

        writer.writerows(history)

    print(
        "Training history saved to models/history.csv"
    )

    print()
    print("Training completed.")


if __name__ == "__main__":
    main()