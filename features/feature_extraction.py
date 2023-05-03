from extractors.lbp import LocalBinaryPatterns
import numpy as np


def save_features(X):
    pass

def extract_features(stacks, transforms=None, extractors=None, save=True):
    """Stack different classes, apply transforms and features. """
    
    # TODO: Concat stacks first
    num_samples = len(stacks)
    num_features = len(extractors)

    # Initialize target matrix.
    y = stacks[:, 1]
    # Get images.
    imgs = stacks[:, 0]
    # Initialize feature matrix.
    X = np.empty(shape=(num_samples, num_features), dtype=np.float128)
    
    for i, feature_extractor in enumerate(extractors):
        X[:, i] = feature_extractor(imgs)
    
    if save:
        save_features(X)
    return X, y


if __name__ == "__main__":
    extractors = [LocalBinaryPatterns]

    stacks = [1 ,2 ,3]
    extract_features(stacks, features=extractors, save=False)
