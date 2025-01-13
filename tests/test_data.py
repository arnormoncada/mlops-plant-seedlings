from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import torch


from src.plant_seedlings.data import plant_seedlings

DATA_LEN = 5539
NUM_CLASSES = 12


def test_my_dataset():
    """Test the MyDataset class."""
    train_loader, val_loader = plant_seedlings(data_path="data/processed")

    # Concatenate the train and validation sets
    combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])

    # Assert that the dataset is an instance of the Dataset class
    assert isinstance(combined_dataset, Dataset), f"Expected Dataset, but got {type(combined_dataset)}"

    # Assert that the dataset has the correct length
    assert len(combined_dataset) == DATA_LEN, f"Expected dataset length {DATA_LEN}, but got {len(combined_dataset)}"

    # Assert that the dataset has the correct number of classes
    original_classes = train_loader.dataset.dataset.classes
    assert len(original_classes) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, but got {len(original_classes)}"

    # Assert that each point in the dataset has the shape (3, 224, 224)
    for i in range(len(combined_dataset)):
        x, y = combined_dataset[i]
        assert x.shape == (3, 224, 224), f"Expected image shape (3, 224, 224), but got {x.shape}"
        assert y >= 0 and y < NUM_CLASSES, f"Expected label in range [0, {NUM_CLASSES}), but got {y}"

    # Assert tht all labels are represented in the dataset
    train_targets = torch.tensor([y for _, y in train_loader.dataset])
    val_targets = torch.tensor([y for _, y in val_loader.dataset])

    # Get the unique labels in the dataset
    train_unique = torch.unique(train_targets)
    assert (train_unique == torch.arange(NUM_CLASSES)).all(), f"Train set does not contain all classes: {train_unique}"

    val_unique = torch.unique(val_targets)
    assert (val_unique == torch.arange(NUM_CLASSES)).all(), f"Validation set does not contain all classes: {val_unique}"
