import torch
import typer
from data import plant_seedlings
from metrics import compute_metrics
import os
import subprocess
from google.cloud import aiplatform
import hydra
import typer.completion
import wandb
from hydra import compose, initialize
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from google.cloud import storage
app = typer.Typer()

# @hydra.main(config_path="config", config_name="config.yaml")



def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Upload a file to GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print(f"Uploading file to {bucket}")
    blob = bucket.blob(destination_blob_name)
    print(f"blob: {blob}")
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}")

@app.command()
def train_VAI() -> None:
    """Train a model on plant-seedlings."""

    print("Training...")
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {DEVICE}")
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config.yaml")

    print(OmegaConf.to_yaml(cfg))

    hparams = cfg.training
    model = hydra.utils.instantiate(cfg.models).to(DEVICE)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    torch.manual_seed(hparams['seed'])
    print("lr = {}, batch_size = {}, epochs = {}, seed = {}".format(cfg.optimizer["lr"], hparams["batch_size"], hparams["epochs"], hparams["seed"]))

    # first we save the model to a file and then log it to wandb as an artifact
    model_name = OmegaConf.select(cfg, "models.model_name")
    if model_name is None:
        model_name = "custom"

    # Load the dataset
    train_dataloader, val_dataloader = plant_seedlings(data_path="data/processed")

    loss_fn = torch.nn.CrossEntropyLoss()

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(hparams["epochs"]):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % (len(train_dataloader) // 5) == 0:
                print(f"Epoch {epoch+1}, Iteration {i}, Loss: {loss.item()}, Accuracy: {accuracy}")

        else:
            # Add validation loop
            model.eval()
            with torch.no_grad():
                for j, (img, target) in enumerate(val_dataloader):
                    img, target = img.to(DEVICE), target.to(DEVICE)
                    y_pred = model(img)

                    test_loss = loss_fn(y_pred, target)
                    accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

       
    print("Training complete")
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path_and_name = hparams["model_path"] + model_name + f"/{datetime_str}" + ".pth"
    print(f"Saving model to {model_path_and_name}")
    torch.save(model.state_dict(), model_path_and_name)
    metrics = compute_metrics(targets, preds)
    print("Final metrics:", metrics)


    PROJECT_ID = os.environ.get("PROJECT_ID", "dtumlops-447815")
    LOCATION   = os.environ.get("LOCATION", "europe-north1")
    BUCKET_NAME = os.environ.get("BUCKET_NAME", "plant-seedling-models")
    
    # e.g. "gs://my-bucket/my-model-ckpt/model.pt"
    GCS_MODEL_ARTIFACT_URI = f"gs://{BUCKET_NAME}/plant_seedlings_ckpt/model.pt"

    upload_blob(
        bucket_name="plant-seedling-models",
        source_file_name=model_path_and_name,
        destination_blob_name="models"
    )

    print(f"Model file uploaded to {GCS_MODEL_ARTIFACT_URI}")

    # -----------------------------
    # 2) Register model in Vertex AI
    # -----------------------------
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Display name can be anything you like:
    MODEL_DISPLAY_NAME = f"plant-seedlings-model-{datetime_str}"

    # Provide a container for serving predictions (e.g., built-in PyTorch serving container):
    SERVING_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-11:latest"

    # This uploads (registers) the model in Vertex AI “Models”:
    uploaded_model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=f"gs://{BUCKET_NAME}/plant_seedlings_ckpt/",
        serving_container_image_uri=SERVING_CONTAINER_URI,
    )

    print("Model registered in Vertex AI:")
    print("Resource name:", uploaded_model.resource_name)    

if __name__ == "__main__":
    typer.run(train_VAI)
