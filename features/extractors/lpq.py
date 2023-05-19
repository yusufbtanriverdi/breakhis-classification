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
        gray_float = gray.astype(np.float32)
        lbp_descriptor = np.zeros_like(gray, dtype=np.uint8)
        lpq_descriptor = []

        height, width = gray.shape
        range_start = self.radius
        range_end = width - self.radius

        angles = 2 * np.pi * np.arange(self.neighbors) / self.neighbors
        cosines = np.cos(angles)
        sines = np.sin(angles)

        # LBP computation
        for i in range(range_start, height - self.radius):
            for j in range(range_start, range_end):
                center = gray_float[i, j]
                vals = gray_float[i + np.round(self.radius * cosines).astype(int),
                                  j - np.round(self.radius * sines).astype(int)]
                lbp_code = np.sum((vals >= center) * (1 << np.arange(self.neighbors)))
                lbp_descriptor[i, j] = lbp_code

        # LPQ encoding
        offset_range = range(-self.block_size // 2, self.block_size // 2 + 1)
        valid_offsets = [(x_offset, y_offset) for x_offset in offset_range for y_offset in offset_range
                         if x_offset != 0 or y_offset != 0]

        for i in range(range_start, height - self.radius):
            for j in range(range_start, range_end):
                lpq_code = np.zeros((), dtype=np.uint32)

                for x_offset, y_offset in valid_offsets:
                    lpq_code = (lpq_code << 1) | (lbp_descriptor[i + x_offset, j + y_offset] > lbp_descriptor[i, j])

                lpq_descriptor.append(lpq_code)

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


