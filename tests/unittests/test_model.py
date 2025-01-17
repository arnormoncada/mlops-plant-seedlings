from src.plant_seedlings.model import MyAwesomeModel, timm_model
import torch
import pytest


@pytest.mark.parametrize(
    "models", [MyAwesomeModel(), timm_model("mobilenetv3_small_050", 12), timm_model("resnet18", 12)]
)
def test_custom_model(models):
    """Test the available models."""
    model = models

    x = torch.randn(32, 3, 224, 224)
    out = model(x)

    # Assert that the output has the correct shape
    assert out.shape == (32, 12), f"Expected output shape (32, 12), but got {out.shape}"
