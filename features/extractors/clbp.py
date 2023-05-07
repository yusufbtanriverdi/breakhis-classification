import cv2
import numpy as np

class CLBP:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        
    def describe(self, image):
        histograms = []
        for (lower, upper) in self.thresholds:
            thresh = cv2.threshold(image, lower, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.threshold(thresh, upper, 255, cv2.THRESH_BINARY_INV)[1]
            for i in range(1, 9):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i))
                neighbors = cv2.morphologyEx(thresh, cv2.MORPH_HITMISS, kernel)
                hist = cv2.calcHist([neighbors], [0], None, [9], [0, 9])
                hist = cv2.normalize(hist, hist).flatten()
                histograms.extend(hist)
        feature_vector = np.concatenate(histograms)
        feature_vector = np.concatenate([feature_vector, ~feature_vector])
        return feature_vector
    
if __name__ == "__main__":
    image = cv2.imread('/Users/melikapooyan/Downloads/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mu = np.mean(gray)
    sigma = np.std(gray)
    pftas = CLBP([(mu + sigma, mu - sigma), (mu - sigma, 255), (mu, 255)])
    feature_vector = pftas.describe(gray)
    print("PFTAS feature vector:", feature_vector)
