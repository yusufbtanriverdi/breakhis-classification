import cv2 as cv
import numpy as np

class HuMoments():

    def __init__(self):
        pass

    def __str__(self):
        return 'shape'

    def describe(self, img):
        # Extract shape features (using Hu Moments)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        moments = cv.moments(gray_img)
        hu_moments = cv.HuMoments(moments)
        shape_features = -np.sign((hu_moments) * np.log10(np.abs(hu_moments)))
        shape_features = shape_features.reshape(-1)
        return shape_features

if __name__ == '__main__':
    pass