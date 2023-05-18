#from extractors.lbp import LocalBinaryPatterns
#from extractors.glcm import GLCM
# from extractors.orb import ORB
from extractors.lpq import LPQ
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
    
    # Create a dictionary to save feature points related to that extractor.
    dict_ = {"image": fnames, 'label': y}

    # Build up filename.
    filename = feature_dir
    df = pd.DataFrame.from_dict(dict_)

    for extractor in extractors:
        filename += str(extractor)
        for j in tqdm(range(len(imgs))):
            feature_values = extractor.describe(imgs[j])
            for k, value in enumerate(feature_values):  
                df.loc[j, f"{str(extractor)}_{k}"] = value  
        print(df)
    if save:
        filename += '.csv'
        df.to_csv(filename, index=False)

    return fnames, df

if __name__ == "__main__":
    extractors = [LPQ(8, 1)]

    stack  = read_data(root='/Users/melikapooyan/Documents/BreaKHis_v1/breast/', mf='40X', mode='binary',shuffle=False)
    if len(stack) == 0:
        print("Please change data dir!!")
        raise IndexError
    
    mf = '40X'
    fnames, df = extract_features(stack, extractors=extractors, save=True, feature_dir=f'features/all/binary/{mf}/')