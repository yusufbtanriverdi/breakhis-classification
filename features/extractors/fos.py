# N. Aggarwal and R. K. Agrawal, "First and Second Order Statistics Features for Classification of 
# Magnetic Resonance Brain Images," 
# Journal of Signal and Information Processing, Vol. 3 No. 2, 2012, pp. 146-153. doi: 10.4236/jsip.2012.32019.

"""Textural features of an image are represented in terms of four first order statistics (Mean, Variance, Skewness, Kurtosis) 
and five-second order statistics (Angular second moment, Contrast, Correlation, Homogeneity, Entropy). 
Since, second order statistics are functions of the distance d and the orientation, hence, for each second order measure, 
the mean and range of the resulting values from the four directions are calculated. Thus, the number of features extracted 
using first and second order statistics are 14."""

"""The variance of an image is not a matrix itself, but it can be calculated from a matrix of pixel values. 
The local variance distribution of a gray level image can be expressed as a matrix, and the Singular Value Decomposition 
(SVD) can be performed on this matrix to measure the structural similarity of two images for image quality assessment. 
However, the variance itself is a scalar value that represents the spread of pixel values in the image, and it is calculated 
by finding the mean value of the pixel values in the image and then calculating the average of the squared differences between 
each pixel value and the mean value.
"""


"""
Can I find in X, Y axes and somehow combine?
PERPLEXITY
Kurtosis is a statistical parameter that measures the "peakedness" or "flatness" of a distribution. 
In the context of image processing, kurtosis can be used to describe the shape of the histogram of
 pixel values in an image. A high kurtosis indicates that the histogram has a sharp peak and heavy 
 tails, while a low kurtosis indicates that the histogram is flatter and more spread out. The kurtosis 
 of an image can be calculated from its histogram, which shows the frequency of occurrence of each pixel value.
 The histogram can be plotted on the X-axis, with the pixel values ranging from 0 to 255, and the frequency of 
 occurrence plotted on the Y-axis. The kurtosis can then be calculated from the histogram using a formula that 
 takes into account the mean and standard deviation of the pixel values. It is not common to plot the kurtosis 
 on the X and Y axes, but it can be combined with other statistical parameters such as variance and skewness to 
 provide a more complete description of the image distribution.

"""
import numpy as np
from scipy import ndimage
from scipy.stats import kurtosis, skew
from skimage.exposure import histogram

class FOS():

    def __init__(self):
        self.info = "FOS"

    def describe(self, image):
        # Detect keypoints and compute their descriptors
        mean = np.mean(image, axis=(0, 1))
        var = ndimage.variance(image)

        hist = histogram(image)

        skewness = skew(hist)

        kurtosis_ = kurtosis(hist)


        return [mean, var, skewness, kurtosis_]
    
    def get_feature(self, image):
        return np.array(self.describe(image), dtype=np.float64)
    
    def __str__(self):
        return 'fos'