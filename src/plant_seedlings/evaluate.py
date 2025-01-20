from enum import Enum
from model import MyAwesomeModel, timm_model
import torch
import os
from data import plant_seedlings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class ModelEnum(Enum):
    custom = "custom"
    resnet = "resnet18"
    mobilenet = "mobilenetv3_small_050"

def get_latest_checkpoint(architecture: str = "resnet18") -> str:
    """Retrieve the latest checkpoint based on the model architecture."""

    # Models directory has subdirectories for each model
    models_dir = "models"
    model_dir = os.path.join(models_dir, architecture)

    # Get the latest checkpoint (based on the last modified time)
    checkpoints = os.listdir(model_dir)
    checkpoints = [os.path.join(model_dir, checkpoint) for checkpoint in checkpoints]
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)

    return latest_checkpoint


def evaluate(model_checkpoint: str, model: str = "resnet") -> None:
    """Evaluate the model on the test set."""

    print("Evaluating the model on the test set.")
    print(f"Model checkpoint provided: {model_checkpoint}")
    print(f"Model architecture: {model}")

    if model_checkpoint is None:
        model_checkpoint = get_latest_checkpoint(model)
        print("No model checkpoint provided. Using the latest checkpoint.")
        print(f"Latest checkpoint: {model_checkpoint}")

    # Load the model
    if model == "resnet18":
        model = timm_model("resnet18", 12).to(DEVICE)
    elif model == "mobilenetv3_small_050":
        model = timm_model("mobilenetv3_small_050", 12).to(DEVICE)
    else:
        model = MyAwesomeModel().to(DEVICE)
    
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_loader = plant_seedlings(data_path="data/processed")

    # Set the model to evaluation mode
    model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Total number of test images: {total}")
    print(f"Number of correctly classified images: {correct}")
    print(f"Accuracy of the model on the test set: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    evaluate(model_checkpoint=None, model="resnet18")
    # evaluate(model_checkpoint=None, model="mobilenetv3_small_050")
    # evaluate(model_checkpoint=None, model="custom")

