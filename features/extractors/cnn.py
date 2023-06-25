import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import cv2 as cv
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import pandas as pd
import numpy as np

def alter_num_classes(model, num_classes=2):
    last_layer_attr = 'fc'
    last_layer = getattr(model, last_layer_attr)

    num_ftrs = last_layer.in_features
    setattr(model, last_layer_attr, torch.nn.Linear(num_ftrs, num_classes))
    return model

def read_means_and_stds(mf):
    info_means = pd.read_csv('features\mean.csv', index_col='mf')    
    info_stdes = pd.read_csv('features\std.csv', index_col='mf')

    return np.array(info_means.loc[mf, :]), np.array(info_stdes.loc[mf, :])

class ResNet18():
    def __init__(self, num_classes = 2, weights='ImageNet', mf='40X'):
        
        self.weights = weights
        self.mf = mf

        if weights == 'ImageNet':
            pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            pretrained_model = alter_num_classes(pretrained_model, num_classes)
            pretrained_model.load_state_dict(torch.load(weights, weights_only=True))
        self.layer = list(pretrained_model.children())[:-3]

    def __str__(self):
        return 'resnet18'
    
    def describe(self, img):
        # Requires image to be Pillow.
        # Load the image and apply the necessary transformations
        image = Image.fromarray(img).convert('RGB')

        if self.weights:
             mean_per_ch, std_per_ch = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
             mean_per_ch, std_per_ch = read_means_and_stds(self.mf)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_per_ch, std=std_per_ch)
            ])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        feature_vector = torch.flatten(features.squeeze())
        return feature_vector

class AlexNet():
    def __init__(self, weights='ImageNet'):
        
        if weights == 'ImageNet':
            pretrained_model = models.resnet18(pretrained=True)

        self.layer = list(pretrained_model.children())[:-3]
        
    def __str__(self):
        return 'alexnet'
    
    def describe(self, image_path):
        # Requires image to be Pillow.

        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        return features.squeeze()

class DenseNet161():
    def __init__(self, weights='ImageNet', last_layer_ind=-5):
        
        if weights == 'ImageNet':
            pretrained_model = models.densenet161(pretrained=True)
        self.block = list(pretrained_model.children())[0]
        self.layer = self.block[0:last_layer_ind]
        
    def __str__(self):
        return 'densenet161'
    
    def describe(self, image_path):
        # Requires image to be Pillow.

        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        return features.squeeze()


class GoogleNet():
    def __init__(self, num_classes=2, weights='ImageNet', mf='40X'):
        
        self.weights = weights
        self.mf = mf
        
        if weights == 'ImageNet':
            pretrained_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        else:
            pretrained_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
            pretrained_model = alter_num_classes(pretrained_model, num_classes)
            pretrained_model.load_state_dict(torch.load(weights, weights_only=True))

        self.layer = list(pretrained_model.children())[:-3]
        
    def __str__(self):
        return 'googlenet'
    
    def describe(self, img):
        # Requires image to be Pillow.
        if self.weights:
             mean_per_ch, std_per_ch = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
             mean_per_ch, std_per_ch = read_means_and_stds(self.mf)
        # Load the image and apply the necessary transformations
        image = Image.fromarray(img).convert('RGB')
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_per_ch, std=std_per_ch)])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        feature_vector = torch.flatten(features.squeeze())
        return feature_vector

class Inception_V3():
    def __init__(self, weights='ImageNet'):
        
        if weights == 'ImageNet':
            pretrained_model = models.inception_v3(pretrained=True)

        train_nodes, eval_nodes = get_graph_node_names(models.inception_v3())
        # print(eval_nodes)  # to find the name of the node
        return_nodes = {
            # node_name: user-specified key for output dict
            'Mixed_7c.branch_pool.bn': 'Last Layer'}
        

        #self.block = list(pretrained_model.children())[0]
        # self.layer_name = 'branch_pool'
        # print(list(pretrained_model.children())[-4])
        # self.layer = list(list(pretrained_model.children())[:-4])
         #self.layer = self.block[0:last_layer_ind]
        self.layer = create_feature_extractor(pretrained_model, return_nodes=return_nodes)
       
        
    def __str__(self):
        return 'inception_v3'
    
    def describe(self, image_path):
        # Requires image to be Pillow.

        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        model = self.layer
        model.eval()
        with torch.no_grad():
                features = model(image)
                myFeatures = features['Last Layer']
        print(myFeatures)
        return myFeatures.squeeze()

class ShuffleNet_v2_x1_0():
    def __init__(self, weights='ImageNet', last_layer_ind=-8):
        
        if weights == 'ImageNet':
            pretrained_model = models.shufflenet_v2_x1_0(pretrained=True)
        self.block = list(pretrained_model.children())[0]
        self.layer = self.block[0:last_layer_ind]
        
    def __str__(self):
        return 'shufflenet_v2_x1_0'
    
    def describe(self, image_path):
        # Requires image to be Pillow.

        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        return features.squeeze()

class SqueezeNet1_0():
    def __init__(self, weights='ImageNet', last_layer_ind=-3):
        
        if weights == 'ImageNet':
            pretrained_model = models.squeezenet1_0(pretrained=True)
        self.block = list(pretrained_model.children())[0]
        self.layer = self.block[0:last_layer_ind]
        
    def __str__(self):
        return 'squeezenet1_0'
    
    def describe(self, image_path):
        # Requires image to be Pillow.

        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        return features.squeeze()

class Vgg16_Bn():
    def __init__(self, weights='ImageNet', last_layer_ind=-6):
        
        if weights == 'ImageNet':
            pretrained_model = models.vgg16_bn(pretrained=True)
        self.block = list(pretrained_model.children())[0]
        self.layer = self.block[0:last_layer_ind]
        
    def __str__(self):
        return 'vgg16_bn'
    
    def describe(self, image_path):
        # Requires image to be Pillow.

        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        return features.squeeze()

class Vgg16():
    def __init__(self, weights='ImageNet', last_layer_ind=-9):
        
        if weights == 'ImageNet':
            pretrained_model = models.vgg16(pretrained=True)
        self.block = list(pretrained_model.children())[0]
        self.layer = self.block[0:last_layer_ind]
        
    def __str__(self):
        return 'vgg16'
    
    def describe(self, image_path):
        # Requires image to be Pillow.

        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        return features.squeeze()

class Vgg19_bn():
    def __init__(self, weights='ImageNet', last_layer_ind=-3):
        
        if weights == 'ImageNet':
            pretrained_model = models.vgg19_bn(pretrained=True)
        self.block = list(pretrained_model.children())[0]
        self.layer = self.block[0:last_layer_ind]
        
    def __str__(self):
        return 'vgg19_bn'
    
    def describe(self, image_path):
        # Requires image to be Pillow.

        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        return features.squeeze()

class Vgg19():
    def __init__(self, weights='ImageNet', last_layer_ind=-4):
        
        if weights == 'ImageNet':
            pretrained_model = models.vgg19(pretrained=True)
        self.block = list(pretrained_model.children())[0]
        self.layer = self.block[0:last_layer_ind]
        
    def __str__(self):
        return 'vgg19'
    
    def describe(self, image_path):
        # Requires image to be Pillow.

        # Load the image and apply the necessary transformations
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize the image to 224x224
                transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Normalize( mean=[0.485, 0.456, 0.406],  # Normalize the image channel-wise
                                        std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        model = nn.Sequential(*self.layer)
        model.eval()
        with torch.no_grad():
                features = model(image)

        return features.squeeze()


if __name__ == "__main__":
    image_path = "C:/Users/hadil/Documents/projects/Machine Learning/project/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-001.png"
    pretrained_model = models.resnet18(pretrained=True)

    obj = ResNet18('ImageNet', -3)
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