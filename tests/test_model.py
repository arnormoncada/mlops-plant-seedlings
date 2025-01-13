from src.plant_seedlings.model import MyAwesomeModel
import torch


def test_custom_model():
    """Test the CustomModel class."""
    model = MyAwesomeModel()

    x = torch.randn(32, 3, 224, 224)
    out = model(x)

    # Assert that the output has the correct shape
    assert out.shape == (32, 12), f"Expected output shape (32, 12), but got {out.shape}"


# TODO: Add test for resnet model
