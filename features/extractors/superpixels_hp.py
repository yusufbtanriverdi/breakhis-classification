# 
import cv2 as cv
import numpy as np

class SuperpixelsEx:
    def __init__(self, algorithm='SLIC'):
        self.algorithm = algorithm
        self.str_to_algorithm = {
            'SLIC': cv.ximgproc.SLIC,
            'SLICO': cv.ximgproc.SLICO,
            'MSLIC': cv.ximgproc.MSLIC
        }

    def superpixels(self, img):
        img = cv.GaussianBlur(img, (3, 3), 0)
        algorithm = self.str_to_algorithm[self.algorithm]
        slic = cv.ximgproc.createSuperpixelSLIC(img, algorithm, 100)
        slic.iterate(10)
        mask = slic.getLabelContourMask()
        img_superpixeled = img.copy()
        img_superpixeled[mask != 0] = (0, 255, 255)
        labels = slic.getLabels()
        img_clustered = np.zeros_like(img)
        num_superpixels = slic.getNumberOfSuperpixels()
        print(num_superpixels)
        for k in range(num_superpixels):
            class_mask = (labels == k).astype("uint8")
            mean_color = cv.mean(img, class_mask)
            img_clustered[class_mask != 0, :] = mean_color[:3]
        return img_clustered, img_superpixeled, labels, num_superpixels

    def describe(self, img):
        # Extract color features (mean values for each channel)
        color_features = np.mean(img, axis=(0, 1))
        # Extract shape features (using Hu Moments)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        moments = cv.moments(gray_img)
        hu_moments = cv.HuMoments(moments)
        shape_features = -np.sign((hu_moments) * np.log10(np.abs(hu_moments)))
        shape_features = shape_features.reshape(-1)
        return np.concatenate([color_features, shape_features])

if __name__ == '__main__':
    image_path = '/Users/melikapooyan/Documents/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png'
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2Lab)

    my_superpixels = SuperpixelsEx(algorithm='SLIC')
    img_clustered, img_superpixeled, labels, num_superpixels = my_superpixels.superpixels(image)

    all_features = []
    for k in range(num_superpixels):
        class_mask = (labels == k).astype("uint8")
        img_clustered_masked = cv.bitwise_and(img_clustered, img_clustered, mask=class_mask)
        features = my_superpixels.describe(img_clustered_masked)
        all_features.append(features)

    all_features = np.array(all_features)

