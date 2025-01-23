import torch
import typer
from data import create_dataloaders, Sweep_preprocess
import os
import hydra
import typer.completion
from hydra import compose, initialize
import wandb
import time
from omegaconf import OmegaConf
from datetime import datetime
from model import timm_model  # Import the custom model class
#from torch.utils.data import DataLoader, random_split
from google.cloud import storage
app = typer.Typer()

def evaluate_model(model, test_dataloader, loss_fn, device, wandb) -> None:
    """
    Evaluate the trained model on the test dataset and log metrics to W&B.

    Args:
        model: The trained model.
        test_dataloader: DataLoader for the test dataset.
        loss_fn: Loss function used for evaluation.
        device: Device to perform evaluation on (CPU/GPU).
        wandb: W&B instance for logging metrics.

    Returns:
        None
    """
    print("Evaluating model on test dataset...")

    model.eval()
    test_loss, test_accuracy = 0.0, 0.0
    total_test_samples = 0
    inference_times = []

    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(device), target.to(device)

            # Measure inference time
            start_time = time.time()
            y_pred = model(img)
            inference_times.append(time.time() - start_time)

            # Calculate loss and accuracy
            loss = loss_fn(y_pred, target)
            test_loss += loss.item() * img.size(0)
            test_accuracy += (y_pred.argmax(dim=1) == target).sum().item()
            total_test_samples += img.size(0)

    # Calculate average metrics
    test_loss /= total_test_samples
    test_accuracy /= total_test_samples
    avg_inference_time = sum(inference_times) / len(inference_times)

    # Log metrics to W&B
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "avg_inference_time": avg_inference_time,
    })

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Avg Inference Time: {avg_inference_time:.6f} seconds")


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
    wandb.init()
    config = wandb.config
    print("Training...")
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {DEVICE}")

    raw_data_path = "data/raw"  # Path to raw dataset

    train_size, val_size, test_size = 0.7, 0.2, 0.1  # Dataset split proportions

    train_dataset, val_dataset, test_dataset = Sweep_preprocess(
        raw_data_path=raw_data_path,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    # Create DataLoaders
    batch_size = config.batch_size  # Get batch size from W&B config
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )  
    # Instantiate the model using the timm_model class
    model = timm_model(
        model_name=config.architecture, 
        num_classes=config.num_classes,
    ).to(DEVICE)

    # Instantiate optimizer directly
    optimizer_cls = getattr(torch.optim, config.optimizer)  # Get optimizer class dynamically
    # Handle optimizer-specific arguments
    if config.optimizer == "SGD":
        optimizer = optimizer_cls(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "Adam":
        optimizer = optimizer_cls(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(config.epochs):
        model.train()
        train_loss, train_accuracy = 0.0, 0.0
        total_train_samples = 0

        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img.size(0)
            train_accuracy += (y_pred.argmax(dim=1) == target).sum().item()
            total_train_samples += img.size(0)

        # Calculate average training loss and accuracy
        train_loss /= total_train_samples
        train_accuracy /= total_train_samples

        # Validation loop
        model.eval()
        val_loss, val_accuracy = 0.0, 0.0
        total_val_samples = 0

        with torch.no_grad():
            for img, target in val_dataloader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                y_pred = model(img)
                loss = loss_fn(y_pred, target)

                val_loss += loss.item() * img.size(0)
                val_accuracy += (y_pred.argmax(dim=1) == target).sum().item()
                total_val_samples += img.size(0)

        val_loss /= total_val_samples
        val_accuracy /= total_val_samples

        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    print("Training complete"),
    # Evaluate the model on the test dataset
    evaluate_model(model, test_dataloader, loss_fn, DEVICE, wandb)

    model_name = f"{wandb.run.name}.pth"
    model_path_and_name = os.path.join("models/", model_name)
    torch.save(model.state_dict(), model_path_and_name)
    print(f"Saving model to {model_path_and_name}")
    wandb.finish()
    # Upload model to GCP bucket
    upload_to_storage(
        bucket_name=os.environ.get("BUCKET_NAME", "plant-seedling-models"),
        source_file_path=model_path_and_name,
        destination_blob_path=model_path_and_name
    )

@app.command()
def sweep():
    """Run a W&B sweep using the sweep configuration."""
    # Load the sweep configuration from sweep1.yaml
    sweep_config_path = "configs/sweeps/sweep1.yaml"
    with open(sweep_config_path, "r") as file:
        sweep_config = OmegaConf.load(file)
    # Convert DictConfig to a standard Python dictionary
    sweep_config_dict = OmegaConf.to_container(sweep_config, resolve=True)
    sweep_id = wandb.sweep(sweep_config_dict, project=sweep_config_dict["project"], entity=sweep_config_dict["entity"])
    wandb.agent(sweep_id, function=train_VAI,count=50)

if __name__ == "__main__":
    typer.run(sweep)
