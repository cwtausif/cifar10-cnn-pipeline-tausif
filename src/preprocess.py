from pathlib import Path
import yaml
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split, Subset


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    val_size = params["preprocess"]["val_size"]
    seed = params["preprocess"]["seed"]

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # No augmentation for validation/test
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # Dataset with training augmentation
    train_dataset_augmented = CIFAR10(
        root=raw_dir,
        train=True,
        transform=train_transform,
        download=False
    )

    # Same raw training data, but without augmentation for validation
    train_dataset_eval = CIFAR10(
        root=raw_dir,
        train=True,
        transform=eval_transform,
        download=False
    )

    # Test dataset without augmentation
    test_dataset = CIFAR10(
        root=raw_dir,
        train=False,
        transform=eval_transform,
        download=False
    )

    # Create reproducible train/validation split
    val_count = int(len(train_dataset_augmented) * val_size)
    train_count = len(train_dataset_augmented) - val_count

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(train_dataset_augmented),
        generator=generator
    ).tolist()

    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    # Training subset uses augmentation
    train_dataset = Subset(
        train_dataset_augmented,
        train_indices
    )

    # Validation subset uses NO augmentation
    val_dataset = Subset(
        train_dataset_eval,
        val_indices
    )

    # Save processed datasets
    torch.save(
        train_dataset,
        processed_dir / "train.pt"
    )

    torch.save(
        val_dataset,
        processed_dir / "val.pt"
    )

    torch.save(
        test_dataset,
        processed_dir / "test.pt"
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("Processed datasets saved to data/processed/")


if __name__ == "__main__":
    main()