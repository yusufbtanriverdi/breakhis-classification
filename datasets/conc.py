import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import cv2

from classifiers.stack import read_data

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

# Import other necessary modules
from torchvision import transforms

def concatenate(stacks, save=True, feature_dir="datasets/all/"):
    """Concatenate images from two folders and save the results."""
    # Create a new list to store the concatenated images.
    concatenated_imgs = []

    for img_A, img_B in tqdm(stacks, desc="Concatenating images"):
        # Load the images from folders A and B
        image_A = cv2.cvtColor(cv2.imread(img_A), cv2.COLOR_BGR2RGB)
        image_B = cv2.imread(img_B, cv2.IMREAD_GRAYSCALE)

        # Concatenate the images
        concatenated_image = np.concatenate((image_A, image_B), axis=1)

        # Append the concatenated image to the list
        concatenated_imgs.append(concatenated_image)

    # Save the concatenated images
    if save:
        os.makedirs(feature_dir, exist_ok=True)
        for i, img in tqdm(enumerate(concatenated_imgs), desc="Saving concatenated images"):
            filename = os.path.join(feature_dir, f"concatenated_{i}.png")
            cv2.imwrite(filename, img)

    return concatenated_imgs


if __name__ == "__main__":
    mf = '40X'
    root_list = ["C:/Users/hadil/Documents/projects/Machine Learning/project/breast/", "C:/Users/hadil/Documents/projects/Machine Learning/project/hog/"]
    stack = read_data(root_list, mf=mf, mode='binary', shuffle=False)

    if len(stack) == 0:
        print("Please provide valid image directories!")
        raise FileNotFoundError

    fnames, df = concatenate(stack, save=True, feature_dir=f'datasets/all/{mf}/')
