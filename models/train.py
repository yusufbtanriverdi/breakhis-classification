import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from models.utilities.losses import FocalLoss
from models.retinanet import RetinaNet
from tools import BreaKHis


print("Hello User! Dataset is loading....")
startTime = time.time()
myDataset = BreaKHis(
                transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]))
print("Elapsed time in min: ", (time.time() - startTime)/60)

print("Size of dataset", len(myDataset))

generator = torch.Generator().manual_seed(42)

training_data, test_data = random_split(myDataset, [0.7, 0.3], generator=generator)
print("Dataset is split for training, validation and test phases --> \n",
        "training:", len(training_data), "\n",
        "test:", (len(test_data)), "\n"
        # "validation:", len(val_data), "\n",
        )


# TODO: Match the distributions of labels.
# TODO: Weighted random sampler has a bug, try to solve it.
# TODO: Implement validation.


BATCH_SIZE = 16
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

model = RetinaNet(num_classes=2)
print("Model summary ---> \n",
      model)

optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      weight_decay=0.0001)

# TODO: Later implement.
#scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     # Milestones are set assuming batch size is 16:
                                     # 60000 / batch_size = 3750
                                     # 80000 / batch_size = 5000
                                     # TODO: Change these according to our # of batches.
#                                     milestones=[3750, 5000],
#                                     gamma=0.1)


# criterion = FocalLoss(2)
criterion = nn.CrossEntropyLoss()

def train(model, cuda=False):

    average_loss = 0

    if cuda:
        model.cuda()
        model = nn.DataParallel(model)

    for batch, (X, y) in enumerate(train_loader):

        optimizer.zero_grad()

        if cuda:
            X.cuda()
            y.cuda()

        X = Variable(X, requires_grad=True)
        yhat = model(X)

        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        # Overcoming: " warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        # scheduler.step()

        average_loss += loss
        print(f'Batch: {batch}, Loss: {loss}, Average Loss: {average_loss / batch + 1}')


def test(model, cuda=False):
    
    average_loss = 0
    correct = 0
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    if cuda:
        model.cuda()
        model = nn.DataParallel(model)

    for X, y in test_loader:

        if cuda:
            X.cuda()
            y.cuda()

        with torch.no_grad:
            X = Variable(X)
            yhat = model(X)

            loss = criterion(yhat, y)
            average_loss += loss
            correct += (yhat.argmax(1) == y).type(torch.float).sum().item() # calculating number of batches

    average_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {average_loss:>8f} \n")
    return average_loss, correct

def eval(model, num_epochs= 50):
    train_losses = []
    test_losses = []
    accuracy_scores = []

    for t in tqdm(range(num_epochs), desc='Training on Breast Histopathology Dataset', unit='epoch'):
        print(f"Epoch {t+1}\n-------------------------------")
        # Uses %90~47 of memory is used when 16 GB ram exists (max 8GB observed). CPU is %50 busy.
        train_loss = train(model)
        train_loss = train_loss.detach().numpy()
        test_loss, accuracy  = test(model)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracy_scores.append(accuracy)
    print("Done!")

    fig, axs = plt.subplots(1, 2, figsize=(8,8))
    axs = axs.ravel()
    axs[0].plot(range(num_epochs), train_losses, ":r")
    axs[0].plot(range(num_epochs), test_losses, "-b")

    axs[1].plot(range(num_epochs), accuracy_scores)


if __name__ == '__main__':
    eval(model)
