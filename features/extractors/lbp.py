import cv2
import numpy as np
from skimage import feature

class LocalBinaryPatterns:
  def __init__(self, numPoints, radius):
    self.numPoints = numPoints
    self.radius = radius

  def describe(self, image, eps = 1e-7):
    # Compute local binary pattern for uniform patterns
    lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
    # Get histogram of uniform patterns
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))

    # Normalize the histogram
    hist = hist.astype('float')
    hist /= (hist.sum() + eps)

    return hist, lbp

  def get_feature(self, image):
    return np.array(self.describe(image)[0], dtype=np.float64)

if __name__ == "__main__":
  import matplotlib.pyplot as plt
    
  image = cv2.imread('/Users/melikapooyan/Downloads/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png')
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  desc = LocalBinaryPatterns(8, 1)
  hist, lbp = desc.describe(gray)
  print("Histogram of Local Binary Pattern value: {}".format(hist))

  # plt.imshow(gray, cmap="gray")
  # plt.show()
  # plt.imshow(lbp, cmap="gray")
  # plt.show()
