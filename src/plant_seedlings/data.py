from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import json
from hydra import compose, initialize


def plant_seed_preprocess(raw_data_path: str, processed_data_path: str) -> None:
    """Preprocess the plant seed dataset."""

    print("Preprocessing data...")

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=raw_data_path, transform=transform)

    with initialize(config_path="../../configs/training", version_base="1.3"):
        cfg = compose(config_name="training_conf.yaml")

    hparams = cfg.training

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [hparams["train_size"], hparams["val_size"], hparams["test_size"]]
    )

    # Create DataLoaders for the train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    # Save the processed data to disk
    torch.save(train_loader, f"{processed_data_path}/train.pth")
    torch.save(val_loader, f"{processed_data_path}/val.pth")
    torch.save(test_loader, f"{processed_data_path}/test.pth")

    # create dict/json with key as index and value as class name
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open("models/classes.json", "w") as f:
        json.dump(idx_to_class, f, indent=4)

    print("Data preprocessing complete.")


def Sweep_preprocess(raw_data_path: str, train_size: float, val_size: float, test_size: float) -> tuple:
    """
    Preprocess the plant seed dataset and split into train, validation, and test sets.
    Args:
        raw_data_path (str): Path to the raw dataset.
        train_size (float): Proportion of the dataset for training.
        val_size (float): Proportion of the dataset for validation.
        test_size (float): Proportion of the dataset for testing.
    Returns:
        tuple: DataLoader objects for train, validation, and test sets.
    """
    print("Preprocessing data...")

    # Define transformations for the images
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Load the dataset
    dataset = datasets.ImageFolder(root=raw_data_path, transform=transform)

    # Calculate dataset sizes
    total_size = len(dataset)
    train_len = int(total_size * train_size)
    val_len = int(total_size * val_size)
    test_len = total_size - train_len - val_len

    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

    # Create a class-to-index mapping and save it to JSON
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open("models/classes.json", "w") as f:
        json.dump(idx_to_class, f, indent=4)

    print("Data preprocessing complete.")
    return train_dataset, val_dataset, test_dataset


def plant_seedlings(data_path: str) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    "Return the plant seed dataset train.pth and val.pth"
    train_loader = torch.load(f"{data_path}/train.pth")
    val_loader = torch.load(f"{data_path}/val.pth")
    test_loader = torch.load(f"{data_path}/test.pth")
    return train_loader, val_loader, test_loader


def create_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for train, validation, and test sets.
    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Testing dataset.
        batch_size (int): Batch size for the DataLoader.
    Returns:
        tuple: DataLoader objects for train, validation, and test sets.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"

    plant_seed_preprocess(raw_data_path, processed_data_path)

    train_loader, val_loader, test_loader = plant_seedlings(data_path="data/processed")
    num_train_images = len(train_loader.dataset)
    num_val_images = len(val_loader.dataset)
    num_test_images = len(test_loader.dataset)

    print(f"Number of training images: {num_train_images}")
    print(f"Number of validation images: {num_val_images}")
    print(f"Number of test images: {num_test_images}")

    print(f"Total number of images: {num_train_images + num_val_images + num_test_images}")

    print(f"Number of classes: {len(train_loader.dataset.dataset.classes)}")
    print(f"Classes: {train_loader.dataset.dataset.classes}")
