import numpy as np
from tools import BreaKHis
from torchvision import transforms
import pandas as pd
import torch
from PIL import ImageOps, Image
import cv2


def scale_decimal(image):
    image = np.array(image)
    # Scale the image between 0 and 255 for each channel
    scaled_image = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        channel = image[..., c]
        scaled_channel = (channel - np.min(channel)) * (1 / (np.max(channel) - np.min(channel)))
        scaled_image[..., c] = scaled_channel
    return scaled_image

class NormalizeByMeanIm(object):
    """    Apply brightness normalization to the dataset.   """

    def __init__(self, mf='40X', *args, **kwargs):
        # Initialize any required variables or parameters here
        self.mf = mf
        info = pd.read_csv('features\mean.csv', index_col='mf')
        self.means = np.array(info.loc[mf, :])

    def __call__(self, img, *args, **kwds):
        img = torch.transpose(img, 0, -1).numpy()
        img = img - self.means/255
        img = scale_decimal(img)
        img = torch.transpose(torch.from_numpy(img), 0, -1)
        return img
    
class CLAHE(object):
    """     Apply CLAHE Equalization to the dataset.    """
    def __init__(self, *args, **kwargs):
        # Initialize any required variables or parameters here
        pass

    def __call__(self, img, *args, **kwds):
        img = torch.transpose(img, 0, -1).numpy()

        # Convert the image to Lab color space
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Convert to uint8
        lab_image = lab_image.astype(np.uint8)
        # Split the Lab image into L, a, and b channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply CLAHE to the L channel 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_l_channel = clahe.apply(l_channel)

        # Merge the processed L channel with the original a and b channels
        clahe_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))

        # Convert the Lab image back to RGB color space
        clahe_rgb_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)

        img = torch.transpose(torch.from_numpy(clahe_rgb_image), 0, -1)

        return img


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    myEqDataset = BreaKHis(
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        NormalizeByMeanIm(),
                        CLAHE(),
                    ]))
    
    print("Size of dataset and samples --> ", len(myEqDataset), myEqDataset[0][0].shape)

    myDataset = BreaKHis(
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        NormalizeByMeanIm(),
                    ]))
    
    print("Size of dataset and samples --> ", len(myDataset), myDataset[0][0].shape)

    fig, axs = plt.subplots(1,2, figsize=(16,16))

    axs = axs.ravel()

    axs[0].imshow(myDataset[0][0][1, :, :])
    axs[1].imshow(myEqDataset[0][0][1, : ,:])
    plt.tight_layout()
    plt.show()

    