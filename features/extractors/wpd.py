from sklearn.preprocessing import StandardScaler
import pywt
import scipy as sc
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

# We can get inspired by this tutorial but I need to study theory behind this. 
# https://www.kaggle.com/code/pluceroo/new-approach-wavelet-packet-decomposition-in-ml 
if __name__ == "__main__":
     # Signal standardization
    img = imread('./examples/SOB_B_A-14-22549AB-40-019.png')
    gray = rgb2gray(img)
    data_std = StandardScaler().fit_transform(gray)         
    
    # WPD tree
    level = 6
    wptree = pywt.WaveletPacket2D(data=data_std, wavelet='db5', mode='symmetric', maxlevel=level)
    level = wptree.get_level(level, order = "freq")     
    features = []        
    for node in level[31:]:
        coefficients = [sub_node.data for sub_node in node]
        print(np.mean(np.abs(np.array(coefficients).ravel())), np.std(coefficients))
        # Create the heatmap
        fig, axs = plt.subplots(8,8,figsize=(16, 16))
        axs = axs.ravel()
        for i, sub_node in enumerate(node):
            axs[i].imshow(sub_node.data, cmap='hot', interpolation='nearest')
        plt.show()
        break