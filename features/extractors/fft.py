import numpy as np
from skimage import io, color, filters
import matplotlib.pyplot as plt

def fft_features(image):
    # Apply a Gaussian filter to the image to reduce noise
    filtered_image = filters.gaussian(image, sigma=1)

    # Compute the Fourier transform of the filtered image
    f = np.fft.fft2(filtered_image)

    # Shift the zero-frequency component to the center of the spectrum
    fshift = np.fft.fftshift(f)

    # Compute the magnitude spectrum (absolute value) of the Fourier transform
    magnitude_spectrum = np.abs(fshift)

    # Extract the real and imaginary parts of the Fourier coefficients as features
    real_part = np.real(f)
    imaginary_part = np.imag(f)

    # Concatenate the real and imaginary parts into a single feature vector
    features = np.concatenate((real_part.flatten(), imaginary_part.flatten()))

    # Normalize the features
    features /= np.max(features)

    return features

if __name__ == "__main__":

    b = 'D:\\BreaKHis_v1\\benign\SOB\\adenosis\\SOB_B_A_14-22549CD\\40X\\SOB_B_A-14-22549CD-40-007.png'
    m='D:\\BreaKHis_v1\\malignant\\SOB\\lobular_carcinoma\\SOB_M_LC_14-13412\\40X\\SOB_M_LC-14-13412-40-002.png'

    # Load an image
    # Source: https://github.com/parham-ap/cytology_dataset/blob/master/dataset/EDF/frame000.png 
    benign = color.rgb2gray(io.imread(b))
    malign = color.rgb2gray(io.imread(m))

    # Convert the image to grayscale
    # gray_image = color.rgb2gray(image)

    benign_features = fft_features(benign)

    malign_features = fft_features(malign)

    plt.plot(malign_features, 'r')
    plt.plot(benign_features, 'b')
    plt.show()