import cv2
import numpy as np
from skimage import feature
import itertools

class CompletedLocalBinaryPatterns:
    def __init__(self, numNeighbors, radius):
        self.numNeighbors = numNeighbors
        self.radius = radius
        self.numBins = 2 ** (3 * numNeighbors)

    def describe(self, image):
        # Compute global threshold
        thresh = np.mean(image)

        # Compute local binary pattern for center pixels
        lbp_center = (image >= thresh).astype('uint8')

        # Compute local binary pattern for magnitude and sign
        lbp_sign = np.zeros_like(image)
        lbp_mag = np.zeros_like(image)
        for x, y in itertools.product(range(-self.radius, self.radius+1), repeat=2):
            if x == 0 and y == 0:
                continue
            elif x ** 2 + y ** 2 > self.radius ** 2:
                continue

            # Compute magnitude and sign difference
            dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            mag_diff = np.abs(cv2.Sobel(image, cv2.CV_32F, x, y, ksize=3))
            sign_diff = (np.arctan2(dy, dx) - np.arctan2(dy[x+self.radius, y+self.radius], dx[x+self.radius, y+self.radius])) / np.pi

            # Compute local binary pattern for magnitude and sign difference
            lbp_mag += (mag_diff >= np.mean(mag_diff)).astype('uint8') * 2 ** (self.numNeighbors - 1 - len(np.where(mag_diff < np.mean(mag_diff))[0]))
            lbp_sign += (sign_diff >= 0).astype('uint8') * 2 ** (self.numNeighbors - 1 - len(np.where(sign_diff < 0)[0]))

        # Combine center, magnitude and sign patterns
        clbp = np.zeros_like(image, dtype='uint16')
        clbp += np.left_shift(lbp_center, 2 * self.numNeighbors)
        clbp += np.left_shift(lbp_sign, self.numNeighbors)
        clbp += lbp_mag

        # Compute completed local binary pattern histogram
        hist = np.histogram(clbp.ravel(), bins=self.numBins, range=(0, self.numBins))[0]

        # Normalize the histogram
        hist = hist.astype('float')
        hist /= hist.sum()

        return hist

    def get_feature(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.describe(gray)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = cv2.imread('/Users/melikapooyan/Downloads/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png')
    clbp = CompletedLocalBinaryPatterns(24, 5)
    hist = clbp.get_feature(image)
    print("Histogram of Completed Local Binary Pattern value: {}".format(hist))

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
