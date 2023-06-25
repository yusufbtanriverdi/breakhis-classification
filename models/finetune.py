from built_in import *
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms as T
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from models.fpcn import FPCN
from models.utilities.losses import FocalLoss
from utilities.phases import *
import os, sys
import warnings 
from sklearn.model_selection import StratifiedShuffleSplit
from torch import random

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from tools import BreaKHis, plot, read_means_and_stds

def assign_class_weights(labels, normalize=True):
    # Compute the class frequencies
    _, class_counts = np.unique(labels, return_counts=True)

    # Compute the inverse class frequencies
    inv_class_counts = 1.0 / class_counts

    # Normalize the weights
    if normalize:
        inv_class_counts /= np.sum(inv_class_counts)

    print(labels)
    # Assign weights to samples based on their class labels
    weights = inv_class_counts[labels.astype(int)]

    return weights

def set_loaders(myDataset, seed=42, test_split=0.3, bs=16):
    

    # Assuming you have your dataset object named 'myDataset' and the desired test split ratio
    # Get the class labels and total number of samples from your dataset
    class_labels = myDataset.targets
    total_samples = len(myDataset)

    # Perform stratified shuffle split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    train_indices, test_indices = next(splitter.split(range(total_samples), class_labels))

    training_data = Subset(myDataset, train_indices)
    test_data = Subset(myDataset, test_indices)

    # training_data, test_data = random_split(myDataset, [1-test_split, test_split], generator=generator)
    print("Dataset is split for training, validation and test phases --> \n",
            "training:", len(training_data), "\n",
            "test:", (len(test_data)), "\n"
            # "validation:", len(val_data), "\n",
            )

    # TODO: Weighted random sampler has a bug, try to solve it.

    BATCH_SIZE = bs

    train_loader = torch.utils.data.DataLoader(training_data, 
                                            batch_size=BATCH_SIZE, 
                                            pin_memory=False,
                                            shuffle=True,
                                            num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_data, 
                                            batch_size=BATCH_SIZE, 
                                            pin_memory=False,
                                            num_workers=0)

    print("Length of loaders and class distributions---> \n",
        len(train_loader), len(train_loader.dataset), "\n",
        len(test_loader), len(test_loader.dataset), "\n"
        )
    
    labels = list()
    for i in test_data.indices:
        labels.append(myDataset.targets[i])
    
    print("Number of unique labels:", np.unique(labels, return_counts=True))

    return train_loader, test_loader

def to_device(data_loader, device):
    for batch in data_loader:
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        elif isinstance(batch, (list, tuple)):
            batch = [to_device(b, device) for b in batch]
        yield batch


if __name__ == '__main__':

    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    # For reproducibility.
    torch.manual_seed(17)

    # Set the device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    mf = '400X'
    mean_per_ch, std_per_ch = read_means_and_stds(mf = mf)

    print("Hello User! Dataset is loading....")
    startTime = time.time()
    myDataset = BreaKHis(
                    transform = T.Compose([
                    T.ToPILImage(),  # Convert numpy.ndarray to PIL Image
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                ]),
                    mf = mf, 
                    mode = 'binary'
                    )
    
    print("Elapsed time in min: ", (time.time() - startTime)/60)
    print("Size of dataset", len(myDataset))


    # Assuming y contains all targets for the dataset
    _, counts = np.unique(myDataset.targets, return_counts=True)

    # Compute class weights based on the frequency of each class in y
    class_weight_0 = counts[0] / len(myDataset.targets)
    class_weight_1 = counts[1] / len(myDataset.targets)
    class_weights = torch.tensor([class_weight_0, class_weight_1], device=device)
    
    print(_, counts)
    
    train_loader, test_loader = set_loaders(
    myDataset,
    seed=42, 
    test_split=0.3, 
    bs=32)
    
    del myDataset

    # Choose models from list.
    models_ = call_builtin_models(pretrained=True)

    # Call spesifically.
    model = FPCN(2, use_pretrained=True)
    model_name = 'fpcnv2'
    
    models_[model_name] = model

    for model_name, model in models_.items():
        # num_features = model.classifier[6].in_features
        # model.classifier[6] = nn.Sequential(
        #     nn.Linear(num_features, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(1024),
        #     nn.Dropout(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(),
        #     nn.Linear(512, 2)
        # )

        print(model_name, "STARTS!")
        model = model.to(device)  # Move the model to the GPU   

        # Define focal loss to incorporate class imbalance.
        # criterion = FocalLoss(alpha=0.3, gamma=3, weight=class_weights.float())
        # Define the loss function with weights
        criterion = nn.CrossEntropyLoss(weight=class_weights.float())

        optimizer = optim.SGD(model.parameters(),
                        lr=0.01,
                        weight_decay=0.001)

        # Annotation follows: magnification factor; augmentation method; pretrained, model type; optimizer type, learning rate; loss, parameters; batch size, sampling strategy; # of epochs
        eval(model, test_loader, train_loader, optimizer, criterion, device, mean_per_ch, std_per_ch, patch=False ,num_epochs=100, mf=mf, model_name=f"400X_on-air-aug_std_weightrandom_pre-{model_name}_sgde-2e-4_bcew_32bs-strf_100ep")

# binary

# 40X -- fpcnv2, resnet18, googlenet, mnasnet, mobilenet (nonaug ------, aug+++++)
# 100X -- fpcnv2, resnet18, googlenet, mnasnet, mobilenet (nonaug ------, aug+++++)
# 200X -- fpcnv2, resnet18, googlenet, mnasnet, mobilenet (nonaug ------, aug+++++)
# 400X -- fpcnv2, resnet18, googlenet, mnasnet, mobilenet, (nonaug ------, aug-----)