import cv2
import numpy as np
from skimage import feature
#from google.colab.patches import cv2_imshow

class LocalBinaryPatterns:
  # LBP:labels the pixels of an image by thresholding the surrounding pixels and expressing them in binary numbers.
  # returns a grayscale image
  # labeling if the center pixel is greater or smaller than the surrounding pixels

  def __init__(self, numPoints, radius):
    self.numPoints = numPoints
    self.radius = radius

  def describe(self, image, eps = 1e-7):
    lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints+3), range=(0, self.numPoints + 2))

    # Normalize the histogram
    hist = hist.astype('float')
    hist /= (hist.sum() + eps)

    return hist, lbp
  
  def get_feature(self, image):
    return np.array(self.describe(image)[1], dtype=np.float64)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    image = cv2.imread('/Users/melikapooyan/Downloads/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-007.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    desc = LocalBinaryPatterns(24, 8)
    hist, lbp = desc.describe(gray)
    print("Histogram of Local Binary Pattern value: {}".format(hist))

    # contrast = contrast.flatten()
    # dissimilarity = dissimilarity.flatten()
    # homogeneity = homogeneity.flatten()
    # energy = energy.flatten()
    # correlation = correlation.flatten()
    # ASM = ASM.flatten()
    hist = hist.flatten()

    # features = np.concatenate((contrast, dissimilarity, homogeneity, energy, correlation, ASM, hist), axis=0) 
    plt.imshow(gray)
    plt.imshow(lbp)