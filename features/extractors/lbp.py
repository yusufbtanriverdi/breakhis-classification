import cv2
import numpy as np
from skimage import feature

class LocalBinaryPatterns():
  def __init__(self, numPoints, radius):
    self.numPoints = numPoints
    self.radius = radius

  def describe(self, image, eps = 1e-7):
    gray=cv2.cvtColor(np.array(image, dtype=np.uint8),cv2.COLOR_BGR2GRAY)
    # Compute local binary pattern for uniform patterns
    lbp = feature.local_binary_pattern(gray, self.numPoints, self.radius, method="uniform")
    # Get histogram of uniform patterns
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))

    # Normalize the histogram
    hist = hist.astype('float')
    hist /= (hist.sum() + eps)

    return hist

  def get_feature(self, image):
    return np.array(self.describe(image)[0], dtype=np.float64)
  
  def __str__(self):
    return 'lbp'

if __name__ == "__main__":
  import matplotlib.pyplot as plt
    
  image = cv2.imread('./examples/lesion.png')
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  desc = LocalBinaryPatterns(8, 1)
  hist, lbp = desc.describe(gray)
  print("Histogram of Local Binary Pattern value: {}".format(hist))

  # plt.imshow(gray, cmap="gray")
  # plt.show()
  # plt.imshow(lbp, cmap="gray")
  # plt.show()
