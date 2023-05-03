  # TAS gives original parameters, unlike PFTAS which gives a variation without any hardcoded parameters.
  # In order to do this we will use mahotas.features.tas method
  # Syntax : mahotas.features.tas(img)
  # Argument : It takes image object as argument
  # Return : It returns 1-D array 

  # Note: Input image should be filtered or should be loaded as gray
  # In order to filter the image we will take the image object which is numpy.
import numpy as np
from skimage.filters import threshold_otsu
import mahotas as mh
import mahotas.thresholding as mht

class PFTAS:
    def __init__(self, thresh_ranges):
        self.thresh_ranges = thresh_ranges
    
    def describe(self, image):
        # Compute the Otsu threshold value
        mu = mh.otsu(image)

        # Compute the standard deviation of pixels above the threshold
        sigma = np.std(image[image > mu])

        # Binarize the image using the specified threshold ranges
        binary_images = [mh.binary_range(image, mu + sigma, mu - sigma),
                         mh.binary_range(image, mu - sigma, 255),
                         mh.binary_range(image, mu, 255)]

        # Compute the PFTAS features for each channel
        features = []
        for channel in range(3):
            channel_feats = []
            for binary_image in binary_images:
                # Compute the adjacency matrix
                adj_matrix = mh.labeled.adjacency(binary_image, connectivity=8)

                # Compute the histogram of white pixels with i white neighbors
                hist = np.bincount(adj_matrix[binary_image > 0], minlength=9)

                # Normalize the histogram
                norm_hist = hist / np.sum(hist)

                # Append the histogram to the channel feature vector
                channel_feats.extend(norm_hist[1:])

            # Concatenate the channel feature vectors
            features.extend(channel_feats)

        # Concatenate the feature vector and its bitwise negated version
        feature_vector = np.concatenate([features, ~np.array(features)])

        return feature_vector
    
    def get_feature(self, image):
        return np.array(self.describe(image), dtype=np.float64)

import cv2

if __name__ == "__main__":
    image = cv2.imread('/path/to/image.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pftas = PFTAS([(mu + sigma, mu - sigma), (mu - sigma, 255), (mu, 255)])
    feature_vector = pftas.get_feature(gray)
    print("PFTAS feature vector:", feature_vector)



