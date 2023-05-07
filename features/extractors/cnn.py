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

        if mode == 'name':
            # Extract layers from the model.
            layer_inds = model_name_to_layer_inds[self.model_name]
            layer_names = list(self.base_model.children())[layer_inds[0]:layer_inds[1]]

            self.model = nn.Sequential(*layer_names)
        
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


if __name__ == "__main__":
    # image_path = "/path/ ... "
    # pretrained_model = models.name_of_the_NN(pretrained=True)

    # features = extract_dlfeature(image, model)
    pass