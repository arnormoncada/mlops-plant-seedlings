import torch
from torch import nn
import timm
from timm import create_model
# import pytorch_lightning  as pl


class timm_model():
    """Simple resnet model fetched from timm."""

    def __init__(self,
                 model_name: str = "resnet18",
                 num_classes: int = 12,
                 ) -> None:
        super().__init__()

        self.model = create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(40000, 128)
        self.fc2 = nn.Linear(128, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        # x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        # print(x.shape)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    print(timm.list_models('mobilenet*'))
