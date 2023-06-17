"""TODO : Superpixel feature inspection."""

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from cv2 import ximgproc
import cv2
import numpy as np
from torchvision import transforms as T
import torch
import time
from tqdm import tqdm

import os, sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from tools import BreaKHis


 
def divide_images_into_patches(images, targets_y, fnames, patch_size = (256, 256), method=cv2.ximgproc.SLICO, 
                               mode='binary', mf = '40X'):
    # fig, axs = plt.subplots(10, 5, figsize=(12, 6))
    # axs = axs.ravel()
    # count = 0

    for ind, image in tqdm(enumerate(images)):
        
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # Convert image to Lab color space for better superpixel segmentation
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        # Separate the stains from the IHC image
        # hed = rgb2hed(image)
        # null = np.zeros_like(hed[:, :, 0])
        # ihc_h = hed2rgb(np.stack((hed[:, :, 0], null, null), axis=-1))
        # ihc_e = hed2rgb(np.stack((null, hed[:, :, 1], null), axis=-1))
        # ihc_d = hed2rgb(np.stack((null, null, hed[:, :, 2]), axis=-1))
  
        # instance and run SLIC
        slic = cv2.ximgproc.createSuperpixelSLIC(image_lab, method, 100)
        slic.iterate(50)

        # replace original image pixels with superpixels means
        labels = slic.getLabels()

        unique_labels = np.unique(labels)
        for lind, label in enumerate(unique_labels):
            mask = labels == label
            # Calculate the bounding box of the superpixel
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            # Crop the corresponding region from the original image
            patch = image[y:y+h, x:x+w, :]
            patch = cv2.resize(patch, patch_size)
            # if count < 50:
            #     # print(np.nonzero(patch))
            #     axs[count].imshow(patch)
            #     axs[count].axis('off')
            #     count += 1
            # else:
            #     plt.tight_layout()
            #     plt.show()
            key = 'benign' if targets_y[ind] == 0 else 'malignant'

            path = f'features/all/{mode}/{mf}/imagelike/superpixels/patches/{key}/{fnames[ind]}_{lind}.png'
            cv2.imwrite(path, patch)
    return


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # from skimage import io

    # image = img_as_float(io.imread('./examples/SOB_B_A-14-22549AB-40-019.png'))

    # # apply SLIC and extract (approximately) the supplied number
    # # of segments
    # # TODO: Improve (fine the best parameters or change the library to opencv -- the ones that teacher used.)
    # segments = slic(image, n_segments = 50, sigma = 10)
    # # show the output of SLIC
    # fig = plt.figure("Superpixels -- %d segments" % (100))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(image, segments, color=(255, 0, 0)))
    # plt.axis("off")
    # # show the plots
    # plt.show()
    
    # src = cv2.GaussianBlur(image,(5,5),0)
    # # Convert to LAB
    # src_lab = cv2.cvtColor(np.array(src, dtype=np.uint8),cv2.COLOR_BGR2LAB) # convert to LAB
    # cv_slico = ximgproc.createSuperpixelSLIC(np.array(src_lab, dtype=np.uint8), algorithm=ximgproc.SLICO, region_size=32)	
    # cv_slico.iterate(num_iterations=50)
    # print(cv_slico.getNumberOfSuperpixels())

    # plt.imshow(cv_slico.getLabelContourMask())
    # plt.show()
    mf = '40X'

    print("Hello User! Dataset is loading....")
    startTime = time.time()
    myDataset = BreaKHis(
                    transform = T.Compose([
                    T.ToPILImage(),  # Convert numpy.ndarray to PIL Image
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                ]),
                    mf = mf, 
                    mode = 'binary'
                    )
    
    print("Elapsed time in min: ", (time.time() - startTime)/60)
    print("Size of dataset", len(myDataset))

    divide_images_into_patches(myDataset.images, myDataset.targets, myDataset.fnames)
