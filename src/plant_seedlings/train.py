import matplotlib.pyplot as plt
import torch
import typer
from data import plant_seedlings
from metrics import compute_metrics
import os
from dotenv import load_dotenv

# from model import MyAwesomeModel
import hydra
import typer.completion
import wandb
from hydra import compose, initialize
from omegaconf import OmegaConf
from pathlib import Path
# from time import time

# path to the .env file
dotenv_path = Path(__file__).parents[2] / ".env"

load_dotenv(dotenv_path)

# Use the API key from the environment variable
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)
app = typer.Typer()

# @hydra.main(config_path="config", config_name="config.yaml")


@app.command()
def train() -> None:
    """Train a model on plant-seedlings."""

    print("Training day and night")
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="config.yaml")

    print(OmegaConf.to_yaml(cfg))

    hparams = cfg.training
    model = hydra.utils.instantiate(cfg.models).to(DEVICE)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    print("lr = {}, batch_size = {}, epochs = {}".format(cfg.optimizer["lr"], hparams["batch_size"], hparams["epochs"]))

    # first we save the model to a file and then log it to wandb as an artifact
    model_name = OmegaConf.select(cfg, "models.model_name")
    if model_name is None:
        model_name = "custom"

    run = wandb.init(
        project="Test_DeleteLater",
        config={
            "lr": cfg.optimizer["lr"],
            "batch_size": hparams["batch_size"],
            "epochs": hparams["epochs"],
            "model": model_name,
        },
    )
    print("wandb run name: ", run.name)
    run_version = run.name.split("-")[-1]
    run.name = f"{model_name}_{run_version}"
    print("wandb new run name: ", run.name)

    
    train_dataloader, _ = plant_seedlings(data_path="data/processed")
    
    # split the train_dataloader into train and validation with 90/10 split
    train_size = int(0.9 * len(train_dataloader.dataset))
    val_size = len(train_dataloader.dataset) - train_size
    train_dataloader, val_dataloader = torch.utils.data.random_split(train_dataloader, [train_size, val_size])

    

    print("Number of training images: ", len(train_dataloader.dataset))
    loss_fn = torch.nn.CrossEntropyLoss()

    statistics = {"train_loss": [], "train_accuracy": []}
    # wandb.log(statistics)
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

            run.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % (len(train_dataloader) // 5) == 0:
                print(f"Epoch {epoch+1}, Iteration {i}, Loss: {loss.item()}, Accuracy: {accuracy}")

                # add a plot of the input images
                # images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                # wandb.log({"images": images})

                # add a plot of histogram of the gradients
                # grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                # wandb.log({"gradients": wandb.Histogram(grads)})
            # if i >= 5:
            #     break
        else:
            # Add validation loop
            model.eval()
            with torch.no_grad():
                for j, (img, target) in enumerate(val_dataloader):
                    img, target = img.to(DEVICE), target.to(DEVICE)
                    y_pred = model(img)

                    test_loss = loss_fn(y_pred, target)
                    accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                    run.log({"val_loss": test_loss.item(), "val_accuracy": accuracy})
        # add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

        # wandb.log({"roc": wandb.plot.roc_curve(targets, preds,
        #                                        labels=None, classes_to_plot=None)})

        # for class_id in range(10):
        #     one_hot = torch.zeros_like(targets)
        #     one_hot[targets == class_id] = 1
        #     _ = RocCurveDisplay.from_predictions(
        #         one_hot,
        #         preds[:, class_id],
        #         name=f"ROC curve for {class_id}",
        #         plot_chance_level=(class_id == 2),
        #     )

        # wandb.plot({"roc": plt})

    print("Training complete")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(hparams["fig_path"])

    metrics = compute_metrics(targets, preds)
    run.summary["final_accuracy"] = metrics["accuracy"]
    run.summary["final_precision"] = metrics["precision"]
    run.summary["final_recall"] = metrics["recall"]
    run.summary["final_f1"] = metrics["f1"]

    # model_path_and_name = hparams["model_path"] + model_name + ".pth"
    model_path_and_name = hparams["model_path"] + model_name + f"/{run.name}" + ".pth"
    print(f"Saving model at {model_path_and_name}")
    torch.save(model.state_dict(), model_path_and_name)

    artifact = wandb.Artifact(
        name=f"{model_name}",
        type="model",
        description="A model trained to classify plant seedlings images",
        metadata={"accuracy": metrics["accuracy"], "precision": metrics["precision"], "recall": metrics["recall"]},
    )
    print("Adding artifact")
    artifact.add_file(model_path_and_name)
    run.log_artifact(artifact)
    print("Done!")


if __name__ == "__main__":
    typer.run(train)
