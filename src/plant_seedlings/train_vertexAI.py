import torch
import typer
from data import plant_seedlings
from metrics import compute_metrics
import os
import hydra
import typer.completion
from hydra import compose, initialize
from omegaconf import OmegaConf
from datetime import datetime
#from torch.utils.data import DataLoader, random_split
from google.cloud import storage
app = typer.Typer()

# @hydra.main(config_path="config", config_name="config.yaml")

def upload_to_storage(bucket_name: str, source_file_path: str, destination_blob_path: str):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  storage.blob._DEFAULT_CHUNKSIZE = 5 * 1024* 1024  # 5 MB This is for my slow internet connection
  storage.blob._MAX_MULTIPART_SIZE = 5 * 1024* 1024  # 5 MB This is for my slow internet connection
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(destination_blob_path)
  blob.upload_from_filename(source_file_path)
  print(f'The file {source_file_path} is uploaded to GCP bucket path: {destination_blob_path}')
  return None

@app.command()
def train_VAI() -> None:
    """Train a model on plant-seedlings.
        Then upload the model to GCP bucket."""

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
            # Validation loop
            model.eval()
            with torch.no_grad():
                for j, (img, target) in enumerate(val_dataloader):
                    img, target = img.to(DEVICE), target.to(DEVICE)
                    y_pred = model(img)

                    #test_loss = loss_fn(y_pred, target)
                    accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)
    print("Training complete")

    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path_and_name = hparams["model_path"] + model_name + f"/{datetime_str}" + ".pth"
    torch.save(model.state_dict(), model_path_and_name)
    print(f"Saving model to {model_path_and_name}")
    metrics = compute_metrics(targets, preds)
    print("Final metrics:", metrics)

    # Upload model to GCP bucket
    upload_to_storage(
        bucket_name=os.environ.get("BUCKET_NAME", "plant-seedling-models"),
        source_file_path=model_path_and_name,
        destination_blob_path=model_path_and_name
    )

if __name__ == "__main__":
    typer.run(train_VAI)
