import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO:  Add more metrics.

def train(model, train_loader, optimizer, criterion, cuda=False):

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


def test(model, test_loader, criterion, cuda=False):
    
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

def eval(model, test_loader, train_loader, optimizer, criterion, num_epochs= 50):
    train_losses = []
    test_losses = []
    accuracy_scores = []

    for t in tqdm(range(num_epochs), desc='Training on Breast Histopathology Dataset', unit='epoch'):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(model, train_loader, optimizer, criterion, )
        train_loss = train_loss.detach().numpy()
        test_loss, accuracy  = test(model, test_loader, criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracy_scores.append(accuracy)
    print("Done!")

    fig, axs = plt.subplots(1, 2, figsize=(8,8))
    axs = axs.ravel()
    axs[0].plot(range(num_epochs), train_losses, ":r")
    axs[0].plot(range(num_epochs), test_losses, "-b")

    axs[1].plot(range(num_epochs), accuracy_scores)

    # TODO: Save results in a csv. 
