import numpy as np
from sklearn.model_selection import train_test_split

import sys
import os

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)
# print(sys.path)
# Now we can import the tools module
from tools import read_images, binary_paths

def np_one_hot_encoder(y):
    """Convert labels to one hot vectors."""
    v = np.zeros((y.size, y.max() + 1))
    v[np.arange(y.size), y] = 1
    return v


def read_data(root, mf, mode = 'binary', shuffle= True):
    stacks = []
    if mode == 'binary':
        paths = binary_paths(root, mf)
        for i, path in enumerate(paths):
            stack = read_images(path, i)
            stacks.append(stack)
    
    stacks = np.array(stacks, dtype=np.uint8)
    if shuffle:
        np.random.shuffle(stacks)
    return stacks

def stack_data(stacks, transforms=None, features=None):
    """Stack different classes, apply transforms and features. """
    # TODO: Concat stacks first
    num_samples = len(stacks)
    num_features = len(features)

    # Initialize target matrix.
    y = stacks[:, 1]
    # Get images.
    imgs = stacks[:, 0]
    # Initialize feature matrix.
    X = np.empty(shape=(num_samples, num_features), dtype=np.float128)
    
    for i, feature_extractor in enumerate(features):
        X[:, i] = feature_extractor(imgs)

    return X, y

def split_data(X, y, one_hot_vector=False, test_size=0.3):
    """Return X_train, X_test, y_train, y_test for classifiers with given split rate."""
    if one_hot_vector:
        y = np_one_hot_encoder(y)
    # Return X_train, X_test, y_train, y_test.
    # TODO: Test stratify.
    return train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
