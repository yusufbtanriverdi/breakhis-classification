from built_in import *
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms as T
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from models.utilities.losses import FocalLoss
from models.retinanet import RetinaNet
from utilities.phases import *
import os, sys
import warnings 

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from tools import BreaKHis

def set_loaders(transform=T.Compose([
                        T.ToTensor(),
                    ]), seed=42, test_split=0.3, bs=16):
    print("Hello User! Dataset is loading....")
    startTime = time.time()
    myDataset = BreaKHis(
                    transform=transform)
    print("Elapsed time in min: ", (time.time() - startTime)/60)
    print("Size of dataset", len(myDataset))

    generator = torch.Generator().manual_seed(seed)

    training_data, test_data = random_split(myDataset, [1-test_split, test_split], generator=generator)
    print("Dataset is split for training, validation and test phases --> \n",
            "training:", len(training_data), "\n",
            "test:", (len(test_data)), "\n"
            # "validation:", len(val_data), "\n",
            )

    # TODO: Match the distributions of labels.
    # TODO: Weighted random sampler has a bug, try to solve it.
    # TODO: Implement validation.

    BATCH_SIZE = bs
    # For unbalanced dataset we create a weighted sampler                                                                                     
    # weights = torch.DoubleTensor(training_data.dataset.weight)                                       
    # training_sampler = WeightedRandomSampler(weights, len(weights), replacement=False)                     
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, pin_memory=True)   
    # weights = torch.DoubleTensor(test_data.dataset.weight)                                       
    # test_sampler = WeightedRandomSampler(weights, len(weights), replacement=False)    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, pin_memory=True)   
        
    # weights = torch.DoubleTensor(val_data.dataset.weight)                                       
    # val_sampler = WeightedRandomSampler(weights, len(weights))                     

    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE,                              
    #                                                            sampler = val_sampler, pin_memory=True)   

    print("Length of loaders ---> \n",
        len(train_loader), len(train_loader.dataset), "\n",
        len(test_loader), len(test_loader.dataset), "\n"
        )

    # TODO: Later implement.
    #scheduler = lr_scheduler.MultiStepLR(optimizer,
                                        # Milestones are set assuming batch size is 16:
                                        # 60000 / batch_size = 3750
                                        # 80000 / batch_size = 5000
                                        # TODO: Change these according to our # of batches.
    #                                     milestones=[3750, 5000],
    #                                     gamma=0.1)
    return train_loader, test_loader

if __name__ == '__main__':

    models = call_builtin_models(pretrained=True)

    # criterion = FocalLoss(2)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = set_loaders(transform=T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()]), 
                                            seed=42, 
                                            test_split=0.3, 
                                            bs=16)
    
    for model in models.values():
        optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      weight_decay=0.0001)
        
        eval(model, test_loader, train_loader, optimizer, criterion)
