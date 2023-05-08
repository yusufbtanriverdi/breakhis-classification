import numpy as np
from sklearn.model_selection import train_test_split

import sys
import os
from tqdm import tqdm
import pandas as pd

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

def read_features(extractors, root='./features/all/', mode='binary', mf='40X'):
    basedir = root + mode + '/' + mf + '/'
    X = []
    if len(extractors) == 1:
        featuredir = basedir + str(extractors[0]) + '.csv'
        csv = pd.read_csv(featuredir)
        return [csv['image'], csv.iloc[:, 2:], csv['label']]
    

    for extractor in extractors:
        featuredir = basedir + str(extractor) + '.csv'
        csv = pd.read_csv(featuredir)
        for col in csv.columns():
            if str(extractor) in col: 
                X.append(csv[col])
    return csv['image'], np.array(X, dtype=np.float32), csv['label']


def read_data(root, mf, mode = 'binary', shuffle= True):
    if mode == 'binary':
        paths = binary_paths(root, mf)

        stack_0 = read_images(paths[0], 0)
        stack_1 = read_images(paths[1], 1)
    
    stack = np.concatenate([stack_0, stack_1])
    if shuffle:
        np.random.shuffle(stack)
    return stack

def stack_data(stacks, transforms=None, features=None, mode='extract'):
    """Stack different classes, apply transforms and features. """

    if mode not in ['extract', 'get']:
        raise ValueError
    
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
        if mode == 'extract':
            X[:, i] = feature_extractor(imgs)
        else:
            X[:, i] = read_features(feature_extractor, imgs)
    return X, y

def split_data(X, y, one_hot_vector=False, test_size=0.3):
    """Return X_train, X_test, y_train, y_test for classifiers with given split rate."""
    if one_hot_vector:
        y = np_one_hot_encoder(y)
    # Return X_train, X_test, y_train, y_test.
    # TODO: Test stratify.
    return train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)

