import numpy as np
from tools import BreaKHis
from torchvision import transforms
from PIL import Image
import torch 
class Normalize(object):
    """Apply brightness normalization to the dataset. """

    def __init__(self, mean, *args, **kwargs):
        # Initialize any required variables or parameters here
        self.mean = mean

    def __call__(self, img, *args, **kwds):
        print(img.shape)
        img = np.array(img-self.mean)
        print(img.shape)
        return img


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print("NORMALIZED")
        # Calculate the mean value from your dataset class
    myNormDataset = BreaKHis(transform=None)
    mean_value = myNormDataset.mean

    myNormDataset = BreaKHis(
                transform=transforms.Compose([
                        Normalize(mean=mean_value),
                        transforms.ToTensor(),
                    ]))
    

    myDataset = BreaKHis(
                transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]))
    
    print("Size of dataset and samples --> ", len(myDataset), myDataset[0][0].shape)

    print("Size of dataset and samples --> ", len(myNormDataset), myNormDataset[0][0].shape)

    fig, axs = plt.subplots(1,3, figsize=(16,16))
    print(mean_value.shape)

    axs = axs.ravel()

    axs[0].imshow(myDataset[0][0][1])
    axs[1].imshow(myNormDataset[0][0][1])
    axs[2].imshow(mean_value[0][0][1])
    plt.tight_layout()
    plt.show()

    