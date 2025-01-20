# TODO: Add test for resnet model
import os
import time
import pytest
import torch
import wandb
from pathlib import Path
from dotenv import load_dotenv
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf


@pytest.fixture
def model():
    """
    Pytest fixture that:
    1. Logs into W&B (if key available).
    2. Reads the artifact name from an environment variable.
    3. Downloads & loads the model with wandb.init() + run.use_artifact().
    4. Returns the loaded model OR None if unavailable.
    """
    # 1. Log into wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        print("WANDB_API_KEY not found. Skipping W&B login and model download.")
        return None
    wandb.login(key=wandb_api_key)

    # 2. The artifact name must be the FULL reference: "entity/project/artifact:version"
    artifact_name = os.getenv("MODEL_ENTITY")
    if not artifact_name:
        print("MODEL_ENTITY not set. Skipping model download.")
        return None

    return download_and_load_model(artifact_name)

def download_and_load_model(artifact_name: str, logdir="models/"):
    """
    Download a model artifact from W&B, load it using Hydra config, and return the loaded model.
    """
    print(f"[INFO] Downloading artifact via run.use_artifact: {artifact_name}")

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "Test_DeleteLater"),
        entity=os.getenv("WANDB_ENTITY_USER"),  # or hardcode your entity if you prefer
        job_type="download_model"
    )

    # Use artifact (type="model")
    artifact = run.use_artifact(artifact_name, type="model")
    artifact_dir = artifact.download(root=logdir)
    print(f"[INFO] Artifact downloaded to: {artifact_dir}")

    # Find the checkpoint file: if multiple, just pick the first
    file_list = list(Path(artifact_dir).glob("*"))
    if not file_list:
        raise FileNotFoundError("No files found in the artifact directory.")
    model_path = str(file_list[0])
    print(f"[INFO] Model checkpoint file: {model_path}")

    # Initialize Hydra config
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config.yaml")
        print("----- Hydra Config Loaded -----")
        print(OmegaConf.to_yaml(cfg))

        # Instantiate the model from config
        model = hydra.utils.instantiate(cfg.models)

        # Load model weights
        print(f"[INFO] Loading state_dict from {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

    return model

def test_model_speed(model):
    """
    Test the inference speed of the model.
    """
    if model is None:
        pytest.skip("Skipping test because model could not be loaded due to missing WANDB_API_KEY or MODEL_ENTITY.")
        return
    
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(torch.rand(1, 1, 28, 28))
    end = time.time()
    elapsed_time = end - start
    print(f"[INFO] Inference time for 100 runs: {elapsed_time:.4f}s")
    return elapsed_time

if __name__ == "__main__":
    """
    Allow this script to be run directly (for local debugging)
    or used via 'pytest tests/performancetests/test_model.py'.
    """
    # Load .env to get WANDB_API_KEY, MODEL_ENTITY, etc.
    dotenv_path = Path(__file__).parents[2] / ".env"
    load_dotenv(dotenv_path)

    # Prepare model using the same logic as the fixture
    wandb_api_key = os.getenv("WANDB_API_KEY")
    artifact_name = os.getenv("MODEL_ENTITY")
    print(f"[INFO] MODEL_ENTITY: {artifact_name}")

    if wandb_api_key and artifact_name:
        wandb.login(key=wandb_api_key)
        my_model = download_and_load_model(artifact_name)
    else:
        print("WANDB_API_KEY or MODEL_ENTITY not found. Model will be None.")
        my_model = None

    # Do a quick speed test
    if my_model is not None:
        elapsed_time = test_model_speed(my_model)

        # Optional: log to W&B
        run = wandb.init(project="Test_DeleteLater", job_type="test_inference")
        wandb.log({"test_elapsed_time": elapsed_time})
        wandb.finish()
    else:
        print("[WARN] Skipping model test because environment variables are missing.")
