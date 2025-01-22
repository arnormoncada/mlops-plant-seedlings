import numpy as np


def extract_features(images):
    """Extract basic image features from a set of images."""
    features = []
    for img in images:
        avg_brightness = np.mean(img)
        contrast = np.std(img)
        sharpness = np.mean(np.abs(np.gradient(img)))
        features.append([avg_brightness, contrast, sharpness])
    return np.round(np.array(features), 5)
