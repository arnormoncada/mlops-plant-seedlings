from data import plant_seedlings
import matplotlib.pyplot as plt
import json


def visualize_images():
    with open("models/classes.json", "r") as json_file:
        class_translator = json.load(json_file)

    _, _, test_loader = plant_seedlings(data_path="data/processed")

    _, (img, target) = next(enumerate(test_loader))

    nrows, ncols = 5, 5
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 12))
    img = img[: (nrows * ncols)]
    target = [str(x.item()) for x in target[: (nrows * ncols)]]

    for i, ax in enumerate(axs.reshape(-1)):
        ax.imshow(img[i].permute(1, 2, 0))
        ax.set_title(f"Class: {class_translator[target[i]]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_images()
