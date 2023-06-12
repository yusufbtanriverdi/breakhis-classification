from built_in import *
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms as T
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler, random_split, RandomSampler
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

from tools import BreaKHis, plot

def set_loaders(myDataset, seed=42, test_split=0.3, bs=16):
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
    train_loader = torch.utils.data.DataLoader(training_data, 
                                               batch_size=BATCH_SIZE, 
                                               pin_memory=True,
                                               shuffle = True
                                               )   
    
    # weights = torch.DoubleTensor(test_data.dataset.weight)                                       
    # test_sampler = WeightedRandomSampler(weights, len(weights), replacement=False)    
    test_loader = torch.utils.data.DataLoader(test_data, 
                                              batch_size=BATCH_SIZE, 
                                              pin_memory=True)   
    
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
    #                                     gamma=0.1))
    return train_loader, test_loader

def read_means_and_stds(mf):
    info_means = pd.read_csv('features\mean.csv', index_col='mf')    
    info_stdes = pd.read_csv('features\std.csv', index_col='mf')

    return np.array(info_means.loc[mf, :]), np.array(info_stdes.loc[mf, :])


if __name__ == '__main__':

    # For reproducibility.
    torch.manual_seed(17)

    mf = '40X'
    mean_per_ch, std_per_ch = read_means_and_stds(mf = mf)
    # criterion = FocalLoss(2)
    criterion = nn.CrossEntropyLoss()

    print("Hello User! Dataset is loading....")
    startTime = time.time()
    myDataset = BreaKHis(
                    transform=T.Compose(
        [ 
            # For built-in models standardized by their mean.
            T.Normalize(mean = mean_per_ch,
                        std = std_per_ch),
            # For built-in models resize.
            T.Resize(256), T.CenterCrop(224), 
            T.ToTensor(),
            ]),
                    mf = mf, mode = 'binary'
                    )
    
    print("Elapsed time in min: ", (time.time() - startTime)/60)
    print("Size of dataset", len(myDataset))


    
    # Augmentation 
    elastic_transformer = T.Compose(
        [T.ToPILImage(),
         T.ElasticTransform(alpha=100.0), 
         T.ToTensor() ] )
    
    orig_imgs = myDataset.images

    print(len(orig_imgs))

    transformed_imgs = [torch.transpose(elastic_transformer(orig_img), 0, -1).numpy() for orig_img in tqdm(orig_imgs[:500])]
    myDataset.images = np.concatenate([np.transpose(myDataset.images, (0, 2, 1, -1)), transformed_imgs])
    myDataset.targets = np.concatenate([myDataset.targets, myDataset.targets])
    myDataset.fnames = np.concatenate([myDataset.fnames, myDataset.fnames])

    train_loader, test_loader = set_loaders(
    myDataset,
    seed=42, 
    test_split=0.3, 
    bs=16)

    print((len(train_loader) + len(test_loader))*16)
    # plot(transformed_imgs, orig_imgs[:2])

    models = call_builtin_models(pretrained=True)

    for model in models.values()[:2]:
        optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      weight_decay=0.0001)
        
        eval(model, test_loader, train_loader, optimizer, criterion)
