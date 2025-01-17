import torch
from torch import nn
import timm

class timm_model(nn.Module):
    """Simple resnet model fetched from timm."""

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 12,
    ) -> None:
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        # Freeze all layers except the final layer
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify last layer based on model architecture
        if hasattr(self.model, "fc"): # E.g. resnet
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif hasattr(self.model, "classifier"): # E.g. mobilenet
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} does not have a fc or classifier attribute.")


    def forward(self, x):
        x = self.model(x)
        return x
    
    def load_state_dict(self, state_dict, strict = True):
        """Override to load state_dict directly into the inner model."""
        self.model.load_state_dict(state_dict, strict)


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

