# import cv2
# import numpy as np

# class CLBP:
#     def __init__(self, thresholds):
#         self.thresholds = thresholds
        
#     def describe(self, image):
#         histograms = []
#         for (lower, upper) in self.thresholds:
#             thresh = cv2.threshold(image, lower, 255, cv2.THRESH_BINARY)[1]
#             thresh = cv2.threshold(thresh, upper, 255, cv2.THRESH_BINARY_INV)[1]
#             for i in range(1, 9):
#                 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i))
#                 neighbors = cv2.morphologyEx(thresh, cv2.MORPH_HITMISS, kernel)
#                 hist = cv2.calcHist([neighbors], [0], None, [9], [0, 9])
#                 hist = cv2.normalize(hist, hist).flatten()
#                 histograms.extend(hist)
#         feature_vector = np.concatenate(histograms)
#         feature_vector = np.concatenate([feature_vector, ~feature_vector])
#         return feature_vector
    
#     def __str__(self):
#         return 'clbp'
import cv2
import numpy as np

class CLBP:
    def __init__(self, radius=5, neighbors=24):
        self.radius = radius
        self.neighbors = neighbors

    def __str__(self):
        return "clbp"

    def describe(self, image, eps = 1e-7):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        output = np.zeros((height, width), np.uint8)
        for i in range(self.radius, height - self.radius):
            for j in range(self.radius, width - self.radius):
                center = gray[i, j]
                code = 0
                for k in range(self.neighbors):
                    x = i + int(round(self.radius * np.cos(2 * np.pi * k / self.neighbors)))
                    y = j - int(round(self.radius * np.sin(2 * np.pi * k / self.neighbors)))
                    if gray[x, y] > center:
                        code += 1 << k
                output[i, j] = code

        # Get histogram of uniform patterns
        (hist, _) = np.histogram(output.ravel(), bins=np.arange(0, 11), range=(0, 10))

        # Normalize the histogram
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)
        return hist



if __name__ == "__main__":
    image = cv2.imread('/Users/melikapooyan/Documents/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png')
    clbp = CLBP(radius=5, neighbors=24)
    feature_vector = clbp.describe(image)
    print("CLBP feature vector:", feature_vector)
    print(feature_vector.shape)

