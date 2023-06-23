import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


str_to_algorithm = {
    'SLIC': cv.ximgproc.SLIC,
    'SLICO': cv.ximgproc.SLICO,
    'MSLIC': cv.ximgproc.MSLIC
}

class SuperpixelsEx():

    def __init__(self, algorithm='SLIC'):
        
        self.algorithm = str_to_algorithm[algorithm]

    def __str__(self):
        return 'superpixels'

    def describeImage(self,img):

        img = cv.GaussianBlur(img, (3, 3), 0)
        # instance and run SLIC
        slic = cv.ximgproc.createSuperpixelSLIC(img, self.algorithm, 100)
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
        pass

if __name__ == '__main__':
    image = cv.imread('.\examples\SOB_B_A-14-22549AB-40-019.png', cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    my_superpixels = SuperpixelsEx()
    img_clustered, img_superpixeled, labels, num_superpixels = my_superpixels.describeImage(image)
    print(np.unique(labels), num_superpixels) 
    plt.imshow(labels)
    plt.show()
    cv.imwrite("C:/Users/hadil/Documents/projects/Machine Learning/project/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-001.png", labels)
    all_features = []
    for k in range(num_superpixels):
        class_mask = (labels == k).astype("uint8")
        ma_sk = (labels == k).astype(np.uint8)
        img_clustered = cv.bitwise_and(img_superpixeled, img_superpixeled, mask=ma_sk)
        features = my_superpixels.extract_features(img_clustered)
        all_features.append(features)


    all_features = np.array(all_features)
    print(all_features.shape)

    #Just for testing:
    #np.savetxt('all_features.csv', all_features, delimiter=',')