import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import cv2 as cv


image_path = "/path/ ... "
pretrained_model = models.name_of_the_NN(pretrained=True)

def extract_dlfeature(pretrained_model, layer_name, image_arr):
    def extract_features(image_path, model):
        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        #transform = transforms.Compose([
        #    transforms.Resize(256),
        #    transforms.CenterCrop(224),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #])
        #image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Extract features from the model
        layer_name = list(pretrained_model.children())[:-1]

        model = nn.Sequential(*layer_name)
        model.eval()
        with torch.no_grad():
            features = model(image)

        
        return features.squeeze()

    features = extract_features(image_path, model)
    layer = pretrained_model.layer(layer_name)
    feature_arr = list()
    for im in image_arr:
        features = extract_features(layer, im)
        feature_arr.append(features)
    return feature_arr
features = extract_features(image, model)