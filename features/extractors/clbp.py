import cv2
import numpy as np

class CLBP:
    def __init__(self, radius=5, neighbors=24):
        self.radius = radius
        self.neighbors = neighbors

    def __str__(self):
        return "clbp"

    def describe(self, image, eps=1e-7):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)  # Resize the image by a factor of 0.5
        height, width = resized.shape
        output = np.zeros((height, width), np.uint8)
        for i in range(self.radius, height - self.radius):
            for j in range(self.radius, width - self.radius):
                center = resized[i, j]
                code = 0
                for k in range(self.neighbors):
                    x = i + int(round(self.radius * np.cos(2 * np.pi * k / self.neighbors)))
                    y = j - int(round(self.radius * np.sin(2 * np.pi * k / self.neighbors)))
                    if resized[x, y] > center:
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

