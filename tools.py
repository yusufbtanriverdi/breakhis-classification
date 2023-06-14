from typing import Any
from torch.utils.data import Dataset
import numpy as np

import cv2
import glob
from tqdm import tqdm 
import pandas as pd 
import os
from PIL import Image
import matplotlib.pyplot as plt

def plot(imgs, orig_imgs, row_title='Transformed Image', **imshow_kwargs):
    num_rows = len(imgs)
    num_cols = 2

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        ax = axs[row_idx, 0]
        ax.imshow(np.asarray(row), **imshow_kwargs)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        ax = axs[row_idx, 1]
        ax.imshow(np.transpose(orig_imgs[row_idx], (1, 0, -1)), **imshow_kwargs)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()


def alter_name(fname):
    fname = fname.split('\\')[-1]
    return fname.split('.')[0]
    
def read_images(path_arr, label, imsize=None):
    # Initialize variables
    min_width = float('inf')
    min_height = float('inf')
    resized_images = []

    # Iterate through all image files in directory
    for filename in tqdm(path_arr):
        if filename.endswith('.png'): # or any other image format
            # Read image
            img = cv2.imread(filename)
            
            # Get image width and height
            height, width, channels = img.shape
            
            # Update minimum width and height
            if width < min_width:
                min_width = width
            if height < min_height:
                min_height = height
            
            # Resize image to minimum width and height
            # resized_img = cv2.resize(img, (min_width, min_height))
            # Let's meta game here:
            
            if imsize:
                resized_img = cv2.resize(img, imsize)   
            else:
                resized_img = cv2.resize(img, (456, 700))   


            fname = alter_name(filename)
                                           
            # Add resized image to list or array
            resized_images.append((resized_img, label, fname))
    
    return resized_images

def read_imageLikefeature(extractor, fnames, root='./features/all/', mode='binary', mf='40X'):
    feature_dir = os.path.join(root, mode, mf, 'imagelike', extractor)
    path = feature_dir + f'/*/*.png'
    path_arr = glob.glob(path)
    print(feature_dir)
    features = []
    for filename in tqdm(path_arr, desc=extractor):
        fname = alter_name(filename)
        features.append([cv2.imread(filename, cv2.IMREAD_GRAYSCALE), fname])

    # Sort by fnames
    features.sort(key=lambda x: list(fnames).index(x[1]))
    # Extract the image arrays
    image_array = np.array([image for image, _ in features])    
    return image_array

def alter_fnames_for_csv(path, save=True):
    df = pd.read_csv(path)
    df['image'] = df['image'].apply(lambda x: alter_name(x))
    if save:
        df.to_csv(path, index=False)
    return df

def binary_paths(root, mf):
    benign = root + f'benign/*/*/*/{mf}/*.png'
    malign = root + f'malignant/*/*/*/{mf}/*.png'
    return glob.glob(benign), glob.glob(malign)


def make_weights_for_balanced_classes(pairs, nclasses):  
    # Source: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3                       
    count = [0] * nclasses 
    for item in pairs:                                                         
        count[item[1]] += 1    
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(pairs)                                              
    for idx, val in enumerate(pairs):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight  

def conc(images, imageLikeFeatures, different_sizes=False):
    # Assume that the order is the same.
    if different_sizes:
        print(images.shape, imageLikeFeatures.shape)
        # Find the minimum size.
        min_height = min(images.shape[1], imageLikeFeatures.shape[1])
        min_width = min(images.shape[2], imageLikeFeatures.shape[2])
        print(min_width, min_height)
        # Rescale the larger images to the minimum size.
        resized_images = []
        for image in images:
            resized_image = cv2.resize(image, (min_width, min_height))
            resized_images.append(resized_image)
        images = np.array(resized_images)

        resized_imageLikeFeatures = []
        for image in imageLikeFeatures:
            resized_image = cv2.resize(image, (min_width, min_height))
            resized_imageLikeFeatures.append(resized_image)
        imageLikeFeatures = np.expand_dims(np.array(resized_imageLikeFeatures), axis=-1)

    # Concatenate the image arrays.
    print(images.shape, imageLikeFeatures.shape)
    images = np.concatenate((images, imageLikeFeatures), axis=-1)
    return images

class BreaKHis(Dataset):
    """TODO [reference_here]``_ Dataset.
    Args:
        root: Base directory for the images.
        labelFile: File directory to target column.
        transform: Transforms to apply on images when calling.
        shuffle: If true, the images will be shuffled.
    """
    def __init__(self, root='../BreaKHis_v1/', mf='40X', mode='binary', transform=None, target_transform = None, shuffle=True, imageLikefeatures=None):
        super(BreaKHis, self).__init__()

        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle

        if mode != 'binary':
            self.nclasses = 4
            print("NOT IMPLEMENTED! Changing mode to binary...")
            mode = 'binary'
        
        if mode == 'binary':
            self.nclasses = 2
            paths = binary_paths(root, mf)
            
            benign_stack = read_images(paths[0], 0)
            malign_stack = read_images(paths[1], 1)

            pairs = np.concatenate([benign_stack, malign_stack])
            
        if shuffle:
            np.random.shuffle(pairs)

        self.images = np.array([image for image in pairs[:, 0]])    
        self.targets = pairs[:, 1]
        self.fnames = pairs[:, -1]
        # self.weight = make_weights_for_balanced_classes(pairs, self.nclasses)
        
        self.imageLikefeatures = imageLikefeatures
        if imageLikefeatures:
            # To make sure it is same size.
            for i, imageLikefeature in enumerate(imageLikefeatures):
                features = read_imageLikefeature(imageLikefeature, self.fnames)
                self.images = conc(self.images, features, different_sizes=True)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
        """

        img = self.images[index]
        target = self.targets[index]

        if self.transform is not None:
            # Convert NumPy array to PIL image
            # img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
if __name__ == '__main__':
    # path = "C:\\Users\\yusuf\\Machine and Deep Learning\\breast_histopathology_clf\\features\\all\\binary\\40X\\pftas.csv"

    # alter_fnames_for_csv(path)

    import time
    from torchvision import transforms
    import torch
    from torch.utils.data import WeightedRandomSampler, random_split

    print("Hello User! Dataset is loading....")
    startTime = time.time()
    myDataset = BreaKHis(
                  transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]),
                        imageLikefeatures=[["hog", 512, 256]])
    
    print("Elapsed time in min: ", (time.time() - startTime)/60)

    print("Size of dataset and samples --> ", len(myDataset), myDataset[0][0].shape)
    print("Let's try to use dataloaders...")
    

    # generator = torch.Generator().manual_seed(42)
    
    # training_data, val_data, test_data = random_split(myDataset, [0.65, 0.25, 0.1], generator=generator)
    # print("Dataset is split for training, validation and test phases --> \n",
    #       "training:", len(training_data),
    #       "validation:", len(val_data),
    #       "test:", (len(test_data)))
    
    # BATCH_SIZE = 16
    # # For unbalanced dataset we create a weighted sampler                                                                                     
    # weights = torch.DoubleTensor(training_data.dataset.weight)
    # print(weights)                                       
    # training_sampler = WeightedRandomSampler(weights, len(weights))                     
    
    # train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE,                              
    #                                                          sampler = training_sampler, pin_memory=True)   
    
    # weights = torch.DoubleTensor(val_data.dataset.weight)  
    # print(weights)                                                                            
    # val_sampler = WeightedRandomSampler(weights, len(weights))                     
    
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE,                              
    #                                                          sampler = val_sampler, pin_memory=True)   
    
    # weights = torch.DoubleTensor(test_data.dataset.weight)                                       
    # print(weights)                                       
    # test_sampler = WeightedRandomSampler(weights, len(weights))    

    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,                              
    #                                                          sampler = test_sampler, pin_memory=True)   
      
    # print("Loaders --> ", len(train_loader), len(val_loader), len(test_loader))

    