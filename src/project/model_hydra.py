import torch
from torch import nn
import hydra


@hydra.main(config_path="config", config_name="config.yaml")
class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, cfg) -> None:
        super().__init__()

        hparams = cfg.model
        self.conv1 = nn.Conv2d(hparams['input_dim'], hparams['conv_size_1'], 3, 1)
        self.conv2 = nn.Conv2d(hparams['conv_size_1'], hparams['conv_size_2'], 3, 1)
        self.conv3 = nn.Conv2d(hparams['conv_size_2'], hparams['conv_size_3'], 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hparams['conv_size_3'], hparams['output_dim'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
