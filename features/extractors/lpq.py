import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift
from skimage.util import view_as_windows

class LocalPhaseQuantization:
    def __init__(self, M=3, Np=8):
        self.M = M
        self.Np = Np

    def __get_local_phase(self, patch):
        # Compute short-term Fourier transform
        f = fft2(patch)
        f = fftshift(f)

        # Get magnitude and phase of Fourier coefficients
        magnitude = np.abs(f)
        phase = np.angle(f)

        # Apply Gaussian weighting to magnitude
        sigma = self.M / 6.4
        weight = np.exp(-(np.arange(-self.M//2, self.M//2+1)**2)/(2*sigma**2))
        weight = np.outer(weight, weight)
        if magnitude.shape != weight.shape:
            h, w = magnitude.shape
            m = (h - weight.shape[0]) // 2
            n = (w - weight.shape[1]) // 2
            magnitude = magnitude[m:m+weight.shape[0], n:n+weight.shape[1]]
        magnitude *= weight

        # Compute the average phase
        sum_sin = np.sum(magnitude * np.sin(phase))
        sum_cos = np.sum(magnitude * np.cos(phase))
        avg_phase = np.arctan2(sum_sin, sum_cos)

        # Convert the phase to [0, 2*pi]
        if avg_phase < 0:
            avg_phase += 2 * np.pi

        return avg_phase

    def __get_binary_code(self, phase):
        # Convert phase to 8-bit integer
        phase = np.uint8(phase * 255 / (2*np.pi))

        # Convert phase to binary code
        binary_code = np.zeros((8,), dtype=np.uint8)
        for i in range(8):
            binary_code[i] = np.uint8((phase >> i) & 1)

        return binary_code

    def describe(self, image):
        # Convert image to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Pad the image with zeros to avoid boundary effects
        pad_size = self.M // 2
        image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)

        # Compute short-term Fourier transform for each patch
        patches = view_as_windows(image, (self.M, self.M), step=1)
        phases = np.zeros(patches.shape[:2])
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                phases[i, j] = self.__get_local_phase(patches[i, j])

        # Compute binary codes for each patch
        binary_codes = np.zeros(patches.shape[:2] + (8,), dtype=np.uint8)
        for i in range(patches.shape[0]):
         for j in range(patches.shape[1]):
          binary_codes[i, j] = self.__get_binary_code(phases[i, j])
        # Accumulate binary codes in histogram
        hist = np.zeros((256,), dtype=np.float32)
        for i in range(binary_codes.shape[0]):
         for j in range(binary_codes.shape[1]):
          idx = int(''.join([str(b) for b in binary_codes[i, j]]), 2)
          hist[idx] += 1

        # Normalize histogram
        hist /= np.sum(hist)

if __name__ == '__main__':
 import matplotlib.pyplot as plt

image = cv2.imread('/path/to/image.png')
lpq = LocalPhaseQuantization()
hist = lpq.describe(image)

# Plot histogram
# plt.bar(range(len(hist)), hist)
#plt.title('LPQ Histogram')
# plt.xlabel('Binary code')
# plt.ylabel('Normalized frequency')
# plt.show()





