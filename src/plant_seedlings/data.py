from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def plant_seed_dataloader(data_path: str, batch_size: int) -> DataLoader:
    """Create a DataLoader for the plant seed dataset."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
