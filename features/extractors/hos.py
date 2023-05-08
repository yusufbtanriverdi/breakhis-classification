# N. Aggarwal and R. K. Agrawal, "First and Second Order Statistics Features for Classification of 
# Magnetic Resonance Brain Images," 
# Journal of Signal and Information Processing, Vol. 3 No. 2, 2012, pp. 146-153. doi: 10.4236/jsip.2012.32019.

"""Textural features of an image are represented in terms of four first order statistics (Mean, Variance, Skewness, Kurtosis) 
and five-second order statistics (Angular second moment, Contrast, Correlation, Homogeneity, Entropy). 
Since, second order statistics are functions of the distance d and the orientation, hence, for each second order measure, 
the mean and range of the resulting values from the four directions are calculated. Thus, the number of features extracted 
using first and second order statistics are 14."""

"""The ASM is a measure of the homogeneity of the image texture, and it represents the sum of the squared elements of the GLCM.
Li X, Hu H, Xiao D, Wang D, Jiang S. Analysis of the spatial distribution of collectors in dust scrubber based on image processing. 
 J Air Waste Manag Assoc. 2019 Jun;69(6):764-777.
 doi: 10.1080/10962247.2019.1586012. Epub 2019 Apr 11. PMID: 30794110.
"""

"""Correlation can also be used to compare different parts of the same image, such as in texture analysis, 
where the correlation between adjacent pixels is used to describe the texture of the image."""

from extractors.glcm import GLCM
import numpy as np
from skimage.filters.rank import entropy
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage.morphology import disk

class HOS:

    def __init__(self):
        self.info = "HOS"

    def describe(self, image):
        # Detect keypoints and compute their descriptors
        
        glcm_vector = GLCM.describe(image)

        asm = np.sum(glcm_vector**2)

        gray = rgb2gray(image)

        entropy_image = entropy(gray, disk(5)) 

        # I am not sure if it should be a matrix?
        scaled_entropy = entropy_image / entropy_image.max()  

        contrast = gray.std()

        return [asm, contrast, scaled_entropy]
    
    def get_feature(self, image):
        return np.array(self.describe(image), dtype=np.float64)

    def __str__(self):
        return 'hos'

