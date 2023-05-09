import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import cv2 as cv


class ResNet():
    def __init__(self, weights='ImageNet', last_layer_ind=-3):
        
        if weights == 'ImageNet':
            pretrained_model = models.resnet18(pretrained=True)

        self.layer = list(pretrained_model.children())[:last_layer_ind]
        
    def __str__(self):
        return 'resnet18'
    
    def describe(self, image_path):
        # Requires image to be Pillow.

        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        return features.squeeze()


if __name__ == "__main__":
    # image_path = "/path/ ... "
    # pretrained_model = models.name_of_the_NN(pretrained=True)

    # features = extract_dlfeature(image, model)
    pass