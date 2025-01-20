# from src.plant_seedlings.model import MyAwesomeModel
import os
import time
import torch
import wandb
from pathlib import Path 
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from dotenv import load_dotenv
import pytest
# TODO: Add test for resnet model
#some comment

@pytest.fixture
def model():
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        print("WANDB_API_KEY not found in the environment. Skipping W&B login.")
        return None

    wandb.login(key=wandb_api_key)

    artifact_name = os.getenv("MODEL_ENTITY")
    return download_and_load_model(artifact_name)

def download_and_load_model(artifact_name: str, logdir="models/"):
    """
    Download a model artifact from W&B, load it using Hydra config, and return the loaded model.
    """
    # 1. Download the artifact
    api = wandb.Api()
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download(root=logdir)

    # 2. Extract the checkpoint file name
    file_name = artifact.files()[0].name
    model_path = os.path.join(artifact_dir, file_name)

    # 3. Initialize Hydra and compose config
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="config.yaml")
        print("----- Hydra Config -----")
        print(OmegaConf.to_yaml(cfg))

        # 4. Instantiate the model from your Hydra config
        model = hydra.utils.instantiate(cfg.models)

        # 5. Load model weights
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

    return model


def test_model_speed(model) -> float:
    """
    Perform a simple inference time test on the model and return elapsed time.
    """
    if model is None:
        pytest.skip("Skipping test because model could not be loaded due to missing WANDB_API_KEY.")
        return
    
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(torch.rand(1, 1, 28, 28))
    end = time.time()
    return end - start


if __name__ == "__main__":
    # 1. Load environment variables from .env
    # path to the .env file
    dotenv_path = Path(__file__).parents[2] / ".env"
    load_dotenv(dotenv_path)

    # 2. Log in to W&B using the key from .env
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        print("WANDB_API_KEY not found in the environment. Skipping W&B login.")
    else:
        wandb.login(key=wandb_api_key)

    # 3. Get the artifact name from env var or default
    artifact_name = os.getenv("MODEL_ENTITY")

    # 4. Download & load the model
    model = download_and_load_model(artifact_name)

    # 5. Test the model speed
    elapsed = test_model_speed(model)
    print(f"[INFO] Inference time for 100 runs: {elapsed:.4f}s")

    # 6. Log the result to W&B
    if wandb_api_key:
        wandb.init(project="Test_DeleteLater", job_type="test")
        wandb.log({"test_elapsed_time": elapsed})
        wandb.finish()

# def test_custom_model():
#     """Test the CustomModel class."""
#     model = MyAwesomeModel()

#     x = torch.randn(32, 3, 224, 224)
#     out = model(x)

#     # Assert that the output has the correct shape
#     assert out.shape == (32, 12), f"Expected output shape (32, 12), but got {out.shape}"
