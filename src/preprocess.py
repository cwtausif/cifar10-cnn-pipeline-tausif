from pathlib import Path
import yaml
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Subset


def main():

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    val_size = params["preprocess"]["val_size"]
    seed = params["preprocess"]["seed"]

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # CIFAR-10 mean and standard deviation
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # ---------------------------------------------------------
    # TRAINING TRANSFORMS
    # ---------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),

        # Mild color augmentation
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(mean, std)
    ])

    # ---------------------------------------------------------
    # VALIDATION / TEST TRANSFORMS
    # ---------------------------------------------------------
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Training dataset with augmentation
    train_dataset_augmented = CIFAR10(
        root=raw_dir,
        train=True,
        transform=train_transform,
        download=False
    )

    # Same data without augmentation for validation
    train_dataset_eval = CIFAR10(
        root=raw_dir,
        train=True,
        transform=eval_transform,
        download=False
    )

    # Test dataset
    test_dataset = CIFAR10(
        root=raw_dir,
        train=False,
        transform=eval_transform,
        download=False
    )

    # ---------------------------------------------------------
    # REPRODUCIBLE TRAIN / VALIDATION SPLIT
    # ---------------------------------------------------------

    val_count = int(len(train_dataset_augmented) * val_size)
    train_count = len(train_dataset_augmented) - val_count

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(train_dataset_augmented),
        generator=generator
    ).tolist()

    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    train_dataset = Subset(
        train_dataset_augmented,
        train_indices
    )

    val_dataset = Subset(
        train_dataset_eval,
        val_indices
    )

    # ---------------------------------------------------------
    # SAVE DATASETS
    # ---------------------------------------------------------

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