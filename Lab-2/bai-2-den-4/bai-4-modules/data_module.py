import os
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Augmentation mạnh hơn
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    test_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, test_tf


def build_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, int, list]:
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")

    train_tf, test_tf = build_transforms(image_size)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=test_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, num_classes, class_names

