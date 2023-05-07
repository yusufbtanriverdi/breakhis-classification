from extractors.lbp import LocalBinaryPatterns
from extractors.glcm import GLCM
from extractors.orb import ORB
from extractors.lpq import LPQ
from extractors.pftas import PFTAS
from extractors.cnn import CNN_extractor
from extractors.fos import FOS
from extractors.hog import HOG
from extractors.hos import HOS
import numpy as np
import pandas as pd

import sys
import os
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)
# print(sys.path)
# Now we can import the tools module

from tools import BreaKHis
from torchvision import transforms

extractors = [LocalBinaryPatterns, GLCM, ORB, LPQ, PFTAS, CNN_extractor, FOS, HOG, HOS]

def save_features(X, filename):
    """Save the extracted features to a CSV file using Pandas."""
    df = pd.DataFrame(X)
    df.to_csv(filename, index=False)

def extract_features(stacks, extractors=None, save=True, filename="features.csv"):
    """Extract features from input images using specified feature extractors."""
    
    # Get number of samples and number of feature extractors.
    num_samples = len(stacks)
    num_features = len(extractors)

    # Initialize target matrix.
    y = np.array(stacks)[:, 1]
    # Get images.
    imgs = np.array(stacks)[:, 0]
    # Initialize feature matrix.
    X = np.empty(shape=(num_samples, num_features), dtype=np.float128)
    
    for i, feature_extractor in enumerate(extractors):
        X[:, i] = feature_extractor(imgs)
    
def save_features(X, filename):
    """Save the extracted features to a CSV file using Pandas."""
    df = pd.DataFrame(X)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    extractors = [LocalBinaryPatterns,GLCM,ORB,LPQ,PFTAS,CNN_extractor,FOS,HOG,HOS]

    myDataset = BreaKHis(
                transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]))
    stacks = [1 ,2 ,3]
    X, y = extract_features(stacks, extractors=extractors, save=True, filename='features.csv')
