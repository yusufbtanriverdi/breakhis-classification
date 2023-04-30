import numpy as np

def stack_data(stacks, transforms=None, features=None, shuffle=True):
    """Stack different classes, apply transforms and features. """
    pass


def prepare_data(stack, one_hot_vector=False):
    """Return X, y for classifiers."""
    X = None
    y = None
    if one_hot_vector:
        y = one_hot_vector_labels(y)
    
    return X, y

def one_hot_vector_labels(y):
    """Return one hot vector labels when required."""
    return y

