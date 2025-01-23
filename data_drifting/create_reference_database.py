import torch
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src/plant_seedlings")
from data import plant_seedlings
from utils import extract_features


# from src.plant_seedlings.data import plant_seedlings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def create_reference_database():
    train_loader, val_loader, test_loader = plant_seedlings(data_path="data/processed")
    loaders = [train_loader, val_loader, test_loader]

    features = []
    for loader in loaders:
        for i, (img, target) in enumerate(loader):
            if (i + 1) % (len(loader) // 5) == 0:
                print(f"Iteration {i + 1} out of {len(loader)}")
            img = img.numpy(force=True)

            targ = np.array([int(x.item()) for x in target])
            feature = extract_features(img)
            comb_feature = np.hstack((feature, targ[:, None]))

            features.append(comb_feature)

    features = np.vstack(features)
    feature_columns = ["Average Brightness", "Contrast", "Sharpness", "Target"]

    feature_df = pd.DataFrame(features, columns=feature_columns)
    feature_df.to_csv("data_drifting/reference_database.csv")


if __name__ == "__main__":
    create_reference_database()
