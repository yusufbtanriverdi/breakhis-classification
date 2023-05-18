#GCLM:calculating how often pairs of pixel with specific values and in a specified spatial relationship occur in an image
#Contrast: Measures the local variations in the gray-level co-occurrence matrix.
#Correlation: Measures the joint probability occurrence of the specified pixel pairs.
#Energy: Provides the sum of squared elements in the GLCM. Also known as uniformity or the angular second moment.
#Homogeneity: Measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal.

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class GLCM:
    def __init__(self, distances, angles, levels):
        self.distances = distances
        self.angles = angles
        self.levels = levels

    def __str__(self):
        return "glcm"
    
    def describe(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Rescale the image
        max_gray_value = np.max(gray)
        scaling_factor = self.levels / max_gray_value
        rescaled_image = (gray * scaling_factor).astype(np.uint8)
        
        # Compute GLCM
        glcm = graycomatrix(rescaled_image, distances=self.distances, angles=self.angles,
                            levels=self.levels, symmetric=True, normed=True)
        
        # Extract GLCM properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        features = []
        for prop in properties:
            prop_values = graycoprops(glcm, prop)
            features.extend(prop_values.ravel())
        
        return features

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load an example image
    image = cv2.imread('/Users/melikapooyan/Documents/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png')

    # Compute GLCM features
    desc = GLCM(distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8)
    features = desc.describe(image)
    
    # Display results
    print(features)


   #plt.imshow(image, cmap='gray')


