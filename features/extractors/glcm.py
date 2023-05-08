#GCLM:calculating how often pairs of pixel with specific values and in a specified spatial relationship occur in an image
#Contrast: Measures the local variations in the gray-level co-occurrence matrix.
#Correlation: Measures the joint probability occurrence of the specified pixel pairs.
#Energy: Provides the sum of squared elements in the GLCM. Also known as uniformity or the angular second moment.
#Homogeneity: Measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal.

import cv2
import numpy as np
import skimage
from skimage.feature import graycomatrix, graycoprops


class GLCM:
    """
    Computes gray-level co-occurrence matrices and extracts features from them.
    """
    def __init__(self, distances, angles, levels=8):
        self.distances = distances
        self.angles = angles
        self.levels = levels

    def describe(self, image):
        # Check if the image needs to be rescaled
        if np.max(image) >= self.levels:
            factor = np.ceil(np.max(image) / self.levels)
            image = (image / factor).astype(np.uint8)
            #gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Compute gray-level co-occurrence matrices
        graycom = graycomatrix(image, self.distances, self.angles, levels=self.levels, symmetric=True, normed=True)

        # Extract features from the matrices
        properties = ['contrast', 'correlation', 'energy', 'homogeneity']
        features = []
        for prop in properties:
            feature_vector = skimage.feature.graycoprops(graycom, prop).flatten()
            features.extend(feature_vector)

        return np.array(features)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load an example image
    image = cv2.imread('/Users/melikapooyan/Downloads/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png', cv2.IMREAD_GRAYSCALE)

    # Compute GLCM features
    desc = GLCM(distances=1, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8)
    features = desc.describe(image)
    
    # Display results
    print(features)
   #plt.imshow(image, cmap='gray')

