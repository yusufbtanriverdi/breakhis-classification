import numpy as np
from tools import BreaKHis
from torchvision import transforms
import pandas as pd
import torch

def scale_decimal(image):
    image = np.array(image)
    # Scale the image between 0 and 255 for each channel
    scaled_image = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        channel = image[..., c]
        scaled_channel = (channel - np.min(channel)) * (1 / (np.max(channel) - np.min(channel)))
        scaled_image[..., c] = scaled_channel
    return scaled_image

class Normalize(object):
    """Apply brightness normalization to the dataset. """

    def __init__(self, mf='40X', *args, **kwargs):
        # Initialize any required variables or parameters here
        self.mf = mf
        info = pd.read_csv('features\mean.csv', index_col='mf')
        self.means = np.array(info.loc[mf, :])

    def __call__(self, img, *args, **kwds):
        img = torch.transpose(img, 0, -1)
        img = img - self.means/255
        img = scale_decimal(img)
        img = torch.transpose(torch.from_numpy(img), 0, -1)
        return img
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt


    myNormDataset = BreaKHis(
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        Normalize(),
                    ]))

    myDataset = BreaKHis(
                transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]))
    
    print("Size of dataset and samples --> ", len(myDataset), myDataset[0][0].shape)
    print("Size of dataset and samples --> ", len(myNormDataset), myNormDataset[0][0].shape)

    fig, axs = plt.subplots(1,2, figsize=(16,16))

    axs = axs.ravel()

    axs[0].imshow(myDataset[0][0][1, :, :])
    axs[1].imshow(myNormDataset[0][0][1, : ,:])
    plt.tight_layout()
    plt.show()

    