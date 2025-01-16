import matplotlib.pyplot as plt
import torch
import typer
from data import plant_seedlings
import os
from dotenv import load_dotenv
# from model import MyAwesomeModel
import hydra
import typer.completion
import wandb
from hydra import compose, initialize
from omegaconf import OmegaConf
from pathlib import Path

# from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
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
    """Train a model on MNIST."""
    
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

    wandb.init(
        project="Test_DeleteLater",
        config={"lr": cfg.optimizer["lr"], "batch_size": hparams["batch_size"], "epochs": hparams["epochs"]},
        name="run",
    )

    train_dataloader, _ = plant_seedlings(data_path="data/processed")

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

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Iteration {i}, Loss: {loss.item()}, Accuracy: {accuracy}")

                # add a plot of the input images
                # images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                # wandb.log({"images": images})

                # add a plot of histogram of the gradients
                # grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                # wandb.log({"gradients": wandb.Histogram(grads)})
            # if i >= 5:
            #     break
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

    # final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    # final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    # final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    # final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    # first we save the model to a file and then log it to wandb as an artifact
    model_name = OmegaConf.select(cfg, "models.model_name")
    if model_name is None:
        model_name = "custom"
    
    model_path_and_name = hparams["model_path"] + model_name + ".pth"
    torch.save(model.state_dict(), model_path_and_name)
    # artifact = wandb.Artifact(
    #     name="corrupt_mnist_model",
    #     type="model",
    #     description="A model trained to classify corrupt MNIST images",
    #     metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    # )
    # artifact.add_file(hparams['model_path'])
    # run.log_artifact(artifact)


if __name__ == "__main__":
    typer.run(train)
