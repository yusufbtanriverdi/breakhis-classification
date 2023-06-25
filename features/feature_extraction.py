from extractors.lbp import LocalBinaryPatterns
from extractors.lpq import LPQ
from extractors.glcm import GLCM
from extractors.orb import ORB
from extractors.pftas import PFTAS
from extractors.clbp import CLBP
from extractors.fos import FOS
from extractors.hog import HOG
from extractors.hos import HOS
from extractors.hog import HOG
from extractors.superpixels import SuperpixelsEx
from extractors.wpd import WPD
from extractors.cnn import GoogleNet, ResNet18

#from extractors.lbp import LocalBinaryPatterns
#from extractors.glcm import GLCM
# from extractors.orb import ORB
#from extractors.lpq import LPQ
# from extractors.pftas import PFTAS
# from extractors.cnn import CNN_extractor
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import cv2
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)
# print(sys.path)
# Now we can import the tools module
import warnings

from classifiers.stack import read_data
from torchvision import transforms

def extract_features(stacks, extractors=None, save=True, feature_dir="features/all/binary/40X/"):
    """Extract features from input images using specified feature extractors."""
    # Initialize target matrix.
    y = np.array(stacks)[:, 1]
    # Get images.
    imgs = np.array(stacks)[:, 0]

    # Get filenames
    fnames = np.array(stacks)[:, 2]

    # # Create a dictionary to save feature points related to that extractor.
    # dict_ = {"image": fnames, 'label': y}

    # # Build up filename.
    # filename = feature_dir
    # df = pd.DataFrame.from_dict(dict_)

    for extractor in extractors:
        for j in tqdm(range(len(imgs))):
            feature_values = list(extractor.describe(imgs[j]))
            feature_values.append(y[j]) 

            filename = feature_dir + str(extractor) + "/" + f'{fnames[j]}.csv'
            # print(filename, len(feature_values))
            np.savetxt(filename, feature_values, delimiter=',')

    # if save:
    #     filename += '.csv'
    #     df.to_csv(filename, index=False)

    return fnames

def assign_multiclass_label_to_features():
    # TODO: From fnames, we can assign labels as new column. 
    pass

# def extract_imageLike(stacks, extractor=None, save=True, feature_dir="features/all/binary/100X/imageLike"):
#      """Extract image-like features from input images using specified feature extractors."""

#      # Initialize target matrix.
#      y = np.array(stacks)[:, 1]
#      # Get images.
#      imgs = np.array(stacks)[:, 0]

#      # Get filenames
#      fnames = np.array(stacks)[:, 2]

#      # Build up filename.
#      # df = pd.DataFrame.from_dict(dict_)
#      features = []
#      pkey = str(extractor) 
#      for j in tqdm(range(len(imgs))):
#          feature_values = extractor.describeImage(imgs[j])
#          features.append(feature_values)
#          label = 'benign' if y[j] == 0 else 'malignant'
#          path = feature_dir + f'/{pkey}/{label}/{fnames[j]}.png'
#          cv2.imwrite(path, feature_values)
#      return fnames, features

if __name__ == "__main__":
    
    mf = '40X'
    extractors = [# LocalBinaryPatterns(8, 1), 
                  # LPQ(radius=3, neighbors=8, block_size=3),
                  # GLCM(distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256),
                  # ORB(num_keypoints=500),
                  # CLBP(radius=5, neighbors=24),
                  # PFTAS()
                  # HOS(),
                  # FOS(),
                  # SuperpixelsEx(),
                  HOG(),
                  # WPD(),
                  # ResNet18(num_classes=2, mf=mf, weights="models/results/40X/weights/40X_on-air-aug_std_none_pre-resnet18_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-23.pth"),
                  # GoogleNet(num_classes=2, mf=mf, weights="models/results/40X/weights/40X_on-air-aug_std_none_pre-googlenet_sgde-2e-4_bcew_32bs-strf_100ep_2023-06-23.pth"),
                  ]

    stack  = read_data(root='../BreaKHis_v1/', mf=mf, mode='binary', shuffle=False)
    if len(stack) == 0:
        print("Please change data dir!!")
        raise NotADirectoryError
    
    fnames, df = extract_features(stack, extractors=extractors, save=True, feature_dir=f'features/all/{mf}/stat/')

    # fnames, fs = extract_imageLike(stack, extractor=extractors[0], save=True, feature_dir=f'D:/imageLike_features/{mf}/')