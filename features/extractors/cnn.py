import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import cv2 as cv

model_name_to_layer_inds = {'resnet18': (0, -1),
                            'resnet52': (0, -1)}

class CNN_extractor():
    def __init__(self, pretrained_model, model_name = 'resnet18', layer_inds=None, mode='name'):

        if mode not in ['layer', 'name']:
            raise ValueError

        if mode == 'layer' and layer_inds == None:
            raise ValueError
        
        self.mode = mode
        self.base_model = pretrained_model
        self.model_name = model_name

        self.layer = list(pretrained_model.children())[:-3]
        
    def __str__(self):
        return str(self.model)
    
    def extract_features(self, image_arr):
         # Add batch dimension
        features_arr = []
        for image in image_arr:
            image = image.unsqueeze(0) 

            # Freeze model and extract features.
            self.model.eval()
            with torch.no_grad():
                features = self.model(image)
            
            features_arr.append(features.squeeze())
            
        return features_arr
    
    def __str__(self):
        return 'cnn'


if __name__ == "__main__":
    image_path = "C:/Users/hadil/Documents/projects/Machine Learning/project/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-001.png"
    pretrained_model = models.resnet18(pretrained=True)

    pass

obj = ResNet('ImageNet', -3)
obj.describe(image_path)
features = obj.describe(image_path)

# Create a 16x32 grid of subplots (to fit all 512 features)
fig, axs = plt.subplots(nrows=16, ncols=16, figsize=(16, 8))

# Loop over the features and plot each one as an image in a subplot
for i in range(len(features)):
    row = i // 16
    col = i % 16
    axs[row, col].imshow(features[i], cmap='gray')
    axs[row, col].axis('off')

# Show the plot
plt.show()