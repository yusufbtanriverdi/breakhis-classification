import cv2
import numpy as np

class LPQ:
    def __init__(self, radius=3, neighbors=8, block_size=3):
        self.radius = radius
        self.neighbors = neighbors
        self.block_size = block_size

    def describe(self, image):
        eps = 1e-7
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = np.zeros(gray.shape, dtype=np.uint8)
        lpq_descriptor = []

        # Step 1: Convert the input image to grayscale if it's not already in grayscale.
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: Divide the image into small overlapping blocks or patches.
        for i in range(self.radius, gray.shape[0] - self.radius):
            for j in range(self.radius, gray.shape[1] - self.radius):
                center = gray[i, j]
                lbp_code = 0

                # Step 3: Compute the LBP code for each pixel in the block.
                for k in range(self.neighbors):
                    angle = 2 * np.pi * k / self.neighbors
                    x = i + int(round(self.radius * np.cos(angle)))
                    y = j - int(round(self.radius * np.sin(angle)))
                    val = gray[x, y]

                    if val >= center:
                        lbp_code |= 1 << k

                # Step 4: Quantize the LBP code to a fixed number of bins.
                lbp[i, j] = lbp_code

        # Step 5: Compute the LPQ code using the LBP codes.
        for i in range(self.radius, gray.shape[0] - self.radius):
            for j in range(self.radius, gray.shape[1] - self.radius):
                lpq_code = 0

                # Step 6: Encode the LPQ code by comparing the LBP codes within the block.
                for x_offset in range(-self.block_size // 2, self.block_size // 2 + 1):
                    for y_offset in range(-self.block_size // 2, self.block_size // 2 + 1):
                        if x_offset != 0 or y_offset != 0:
                            lpq_code <<= 1
                            lpq_code |= lbp[i + x_offset, j + y_offset] > lbp[i, j]

                lpq_descriptor.append(lpq_code)

        # Step 7: Concatenate the LPQ codes from all the blocks to form the final LPQ descriptor.
        hist, _ = np.histogram(lpq_descriptor, bins=np.arange(0, 2**self.block_size + 1), range=(0, 2**self.block_size))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist


if __name__ == "__main__":
    # Load image
    image = cv2.imread('/Users/melikapooyan/Documents/BreaKHis_v1/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-003.png')

    # Compute LPQ descriptor
    lpq = LPQ()
    lpq_desc = lpq.describe(image)

    print(lpq_desc)







