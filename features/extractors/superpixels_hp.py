import cv2 as cv
import numpy as np

image = cv.imread('../path/...', cv.IMREAD_COLOR)
image = cv.cvtColor(image, cv.COLOR_BGR2Lab)

class SuperpixelsEx:

    def superpixels(self,img):
        
        img = cv.GaussianBlur(img, (3, 3), 0)

        # instance and run SLIC
        slic = cv.ximgproc.createSuperpixelSLIC(img, cv.ximgproc.SLIC, 100)
        slic.iterate(10)

        # get and draw superpixels
        mask = slic.getLabelContourMask()
        
        img_superpixeled = img.copy()
        img_superpixeled[mask != 0] = (0, 255, 255)

        # replace original image pixels with superpixels means
        labels = slic.getLabels()

        img_clustered = np.zeros_like(img)

        num_superpixels = slic.getNumberOfSuperpixels()
        print(num_superpixels)
        for k in range(num_superpixels):
            class_mask = (labels == k).astype("uint8")
            mean_color = cv.mean(img, class_mask)
            img_clustered[class_mask != 0, :] = mean_color[:3]

        return img_clustered, img_superpixeled, labels, num_superpixels



    def extract_features(self, img):
        # Extract color features (mean values for each channel)
        color_features = np.mean(img, axis=(0, 1))
        # Extract shape features (using Hu Moments)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        moments = cv.moments(gray_img)
        hu_moments = cv.HuMoments(moments)
        shape_features = -np.sign((hu_moments) * np.log10(np.abs(hu_moments)))
        shape_features = shape_features.reshape(-1)

        return np.concatenate((color_features, shape_features))


my_superpixels = SuperpixelsEx(image)
img_clustered, img_superpixeled, labels, num_superpixels = my_superpixels.superpixels(image)

all_features = []
for k in range(num_superpixels):
    class_mask = (labels == k).astype("uint8")
    ma_sk = (labels == k).astype(np.uint8)
    img_clustered = cv.bitwise_and(super, super, mask=ma_sk)
    features = my_superpixels.extract_features(img_clustered)
    all_features.append(features)

all_features = np.array(all_features)