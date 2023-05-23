from sklearn.preprocessing import StandardScaler
import pywt
import scipy as sc
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

class WPD():
    """
    Computes WPD frequency-banded images for primer and seconder noe at level 6th.
    """
    def __init__(self, level=6):
        self.level = level

    def describe(self, image):
        features = self.describeImage(image)
        # Flatten the array using np.reshape
        flattened_array = np.reshape(features, (-1,))

        # Returns image, should it?
        return flattened_array

    def describeImage(self, image):
        gray = rgb2gray(image)
        data_std = StandardScaler().fit_transform(gray)         
        
        # WPD tree
        level = self.level
        wptree = pywt.WaveletPacket2D(data=data_std, wavelet='db5', mode='symmetric', maxlevel=level)
        level = wptree.get_level(level, order = "freq")     

        primer_node = level[0]
        seconder_node = level[1]
        coefficients_1st = [sub_node.data for sub_node in primer_node]
        coefficients_2nd = [sub_node.data for sub_node in seconder_node]

        # Returns two images.
        return np.array([np.mean(coefficients_1st, axis=0), np.mean(coefficients_2nd, axis=0)])
    
    def __str__(self):
        return 'wpd'

# We can get inspired by this tutorial but I need to study theory behind this. 
# https://www.kaggle.com/code/pluceroo/new-approach-wavelet-packet-decomposition-in-ml 
if __name__ == "__main__":
    import pandas as pd
     # Signal standardization
    img = imread('./examples/SOB_B_A-14-22549AB-40-019.png')
    gray = rgb2gray(img)
    data_std = StandardScaler().fit_transform(gray)         
    
    # WPD tree
    level = 6
    wptree = pywt.WaveletPacket2D(data=data_std, wavelet='db5', mode='symmetric', maxlevel=level)
    level = wptree.get_level(level, order = "freq")     
    features = []        
    for node in level[2:]:
        coefficients = [sub_node.data for sub_node in node]
        print(np.mean(np.abs(np.array(coefficients).ravel())), np.std(coefficients))
        # Create the heatmap
        print(np.array(coefficients).shape)
        print(np.mean(coefficients, axis=0).shape)
        # fig, axs = plt.subplots(8,8,figsize=(16, 16))
        # axs = axs.ravel()
        # for i, sub_node in enumerate(node):
        #     axs[i].imshow(sub_node.data, cmap='hot', interpolation='nearest')
        plt.imshow(np.mean(coefficients, axis=0), cmap='hot')
        plt.colorbar()
        plt.show()
        break

    extractor = WPD()
    feature_values = extractor.describe(img)
    # Create a new DataFrame with the feature values
    # Filter and ignore performance warnings
    # warnings.filterwarnings("ignore")
    column_labels = [f"{str(extractor)}_{i}" for i in range(1, 1+ len(feature_values))]
    print(len(column_labels), len(feature_values))
    new_df = pd.DataFrame(columns=column_labels)

    new_df.loc[0, column_labels] = feature_values
    # Reset warning filters (if necessary)
    # warnings.resetwarnings()
    df = pd.DataFrame(columns=['image', 'label'])
    df.loc[0, 'image'] = 'X'
    df.loc[0, 'label'] = 1
    df = pd.concat([df, new_df], axis=1)
    print(df)