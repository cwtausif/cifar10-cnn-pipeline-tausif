from pathlib import Path

from torchvision.datasets import CIFAR10


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading/loading CIFAR-10...")

    CIFAR10(
        root=raw_dir,
        train=True,
        download=True,
    )

    CIFAR10(
        root=raw_dir,
        train=False,
        download=True,
    )

    print("CIFAR-10 raw dataset is ready in data/raw/")


if __name__ == "__main__":
    main()