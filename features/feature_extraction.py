from extractors.lbp import LocalBinaryPatterns
from extractors.glcm import GLCM
# from extractors.orb import ORB
# from extractors.lpq import LPQ
# from extractors.pftas import PFTAS
# from extractors.cnn import CNN_extractor
# from extractors.fos import FOS
# from extractors.hog import HOG
# from extractors.hos import HOS
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)
# print(sys.path)
# Now we can import the tools module


from classifiers.stack import read_data
from torchvision import transforms

extractors = [GLCM]

def save_features(X, filename):
    """Save the extracted features to a CSV file using Pandas."""
    df = pd.DataFrame(X)
    df.to_csv(filename, index=False)

def extract_features(stacks, extractors=None, save=True, feature_dir="features/all/binary/40X/"):
    """Extract features from input images using specified feature extractors."""
    
    # Get number of samples and number of feature extractors.
    num_samples = len(stacks)
    num_features = len(extractors)

    # Initialize target matrix.
    y = np.array(stacks)[:, 1]
    # Get images.
    imgs = np.array(stacks)[:, 0]

    # Get filenames
    fnames = np.array(stacks)[:, 2]
    # Initialize feature matrix.
    X = []

    for j in tqdm(range(len(imgs))):
        tmp = []
        for i, feature_extractor in enumerate(extractors):
            tmp.append(feature_extractor.describe(imgs[j]))
        X.append(np.array(tmp, dtype=np.float64))
        if save:
            X_with_y = X.copy()
            X_with_y.append(y)
            filename = feature_dir + str(feature_extractor)
            save_features(X_with_y, fnames, title = str(feature_extractor), filename=filename)
    return fnames, X, y

    
def save_features(X, fnames, title, filename):
    """Save the extracted features to a CSV file using Pandas."""
    dicto = {"image": fnames, title: X, 'label': y}
    df = pd.DataFrame.from_dict(dicto)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    extractors = [LocalBinaryPatterns(numPoints=8, radius=1)]

    stack  = read_data(root='D:/BreaKHis_v1/', mf='40X', mode='binary',shuffle=False)
    if len(stack) == 0:
        print("Please change data dir!!")
        raise IndexError
    
    mf = '40X'
    fnames, X, y = extract_features(stack, extractors=extractors, save=True, feature_dir=f'features/all/binary/{mf}')
    
