import cv2
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

class PFTAS:
    def __init__(self):
        pass

    def describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute Otsu's threshold
        thresh = threshold_otsu(gray)

        # Binarize the image using multiple thresholds
        binary1 = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
        binary2 = cv2.threshold(gray, thresh - np.std(gray[gray > thresh]), 255, cv2.THRESH_BINARY)[1]
        binary3 = cv2.threshold(gray, thresh + np.std(gray[gray < thresh]), 255, cv2.THRESH_BINARY_INV)[1]

        # Compute normalized histograms for each thresholded image
        histograms = []
        for binary in [binary1, binary2, binary3]:
            hist, _ = np.histogram(binary, bins=9, range=(0, 256))
            hist = hist.astype('float')
            hist /= (hist.sum() + 1e-7)
            histograms.append(hist)

        # Concatenate histograms for each RGB channel
        features = []
        for channel in cv2.split(image):
            for hist in histograms:
                features.extend(hist)
        features = np.array(features)

        # Convert features array to np.uint8 and concatenate the feature vector with its bitwise negated version
        features = np.concatenate([features.astype(np.uint8), np.bitwise_not(features.astype(np.uint8))])

        return features

    def __str__(self):
        return 'pftas'

if __name__ == '__main__':
    pftas = PFTAS()
    image = cv2.imread('/Users/melikapooyan/Documents/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-005.png')
    features = pftas.describe(image)
    print(features)

