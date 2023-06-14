import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import Accuracy, Recall, AveragePrecision, AUROC, MeanAbsolutePercentageError, F1Score, R2Score, CohenKappa
import pandas as pd
import datetime
import numpy as np

def train(model, train_loader, optimizer, criterion, eval_metrics, device, epoch=-1):

    average_loss = 0
    metric_values = {metric_name: [] for metric_name in eval_metrics.keys()}

    # model.to(device)
    if device != 'cpu':
        model = nn.DataParallel(model.to(device))

    for batch, (X, y) in enumerate(train_loader):

        optimizer.zero_grad()

        X = X.to(device)
        y = y.to(device)

        X = X.requires_grad_()

        yhat = model(X)
        yhat = yhat.to(device)

        loss = criterion(yhat, y)
        
        loss.backward()
        optimizer.step()

        average_loss += loss.item()

        # Compute evaluation metrics
        with torch.no_grad():
             # Convert labels to one-hot vectors and vice-versa.
            y_vectors = torch.eye(2, device=device, dtype=torch.long)[y]
            yhat_labs = torch.argmax(yhat, dim=1).to(device)
            for metric_name, metric in eval_metrics.items():
                try:
                    # Try as one-hot-vectors.
                    # print("I tried THIS!")
                    metric_val = metric(yhat_labs, y)
                except ValueError as e:
                    # Try as logits. 
                    # print("BUT IT DIDNT WORK SO!")
                    metric_val = metric(yhat, y_vectors)
                
                metric_values[metric_name].append(metric_val.item())
        if batch % 10 == 0:
            #  print(f"loss: {loss:>7f}, average loss: {average_loss/len(train_loader):>5f}")
            pass

    average_loss /= len(train_loader)
    epoch_scores = {
        'Epoch': epoch + 1,
        'Average Loss': average_loss,
        **{metric_name: sum(metric_values[metric_name]) / len(metric_values[metric_name]) for metric_name in eval_metrics}
    }

    print(epoch_scores)
    return epoch_scores

def test(model, test_loader, criterion, eval_metrics, device, epoch=-1, mode='binary'):

    average_loss = 0
    metric_values = {metric_name: [] for metric_name in eval_metrics.keys()}

    # model.to(device)
    if device != 'cpu':
        model = nn.DataParallel(model)

    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():

            X = X.requires_grad_()
            yhat = model(X)
            yhat = yhat.to(device)

            loss = criterion(yhat, y)
            average_loss += loss.item()

            # Convert labels to one-hot vectors and vice-versa.
            y_vectors = torch.eye(2, device=device, dtype=torch.long)[y]
            yhat_labs = torch.argmax(yhat, dim=1).to(device)
            for metric_name, metric in eval_metrics.items():
                try:
                    # Try as one-hot-vectors.
                    # print("I tried THIS!")
                    metric_val = metric(yhat_labs, y)
                except ValueError as e:
                    # Try as logits. 
                    # print("BUT IT DIDNT WORK SO!")
                    metric_val = metric(yhat, y_vectors)

                metric_values[metric_name].append(metric_val.item())
        
    average_loss /= len(test_loader)
    epoch_scores = {
        'Epoch': epoch + 1,
        'Average Loss': average_loss,
        **{metric_name: sum(metric_values[metric_name]) / len(metric_values[metric_name]) for metric_name in eval_metrics}
    }

    print(epoch_scores)
    return epoch_scores

def eval(model, test_loader, train_loader, optimizer, criterion, device, num_epochs= 1, mode='binary', model_name=None):

    eval_metrics = {
    'accuracy_score': Accuracy(task=mode).to(device),
    'roc_auc_score': AUROC(task=mode).to(device),
    'average_precision_score' : AveragePrecision(task=mode).to(device),
    'mean_absolute_percentage_error' : MeanAbsolutePercentageError().to(device),
    'f1_score' : F1Score(mode).to(device),
    'r2_score' : R2Score().to(device),
    'recall_Score' : Recall(mode).to(device),
    'cohen_kappa_score' : CohenKappa(mode).to(device)
    }
    # TODO:  Add pattern recognition rate.

    for t in tqdm(range(num_epochs), desc='Training on Breast Histopathology Dataset', unit='epoch'):
        print(f"Epoch {t+1}\n-------------------------------")
        train_scores = train(model, train_loader, optimizer, criterion, eval_metrics, device, epoch = t,)
        test_scores = test(model, test_loader, criterion, eval_metrics, device, epoch= t)
        if t == 0:
            train_df = pd.DataFrame(train_scores, index=[0])
            test_df = pd.DataFrame(test_scores, index=[0])
        else:
            train_df = pd.concat([train_df, pd.DataFrame(train_scores, index=[0])])
            test_df = pd.concat([test_df, pd.DataFrame(test_scores, index=[0])])
    
    # Save the DataFrame as a CSV file
    if not model_name:
        model_name = model.__class__.__name__

    # Get the current date
    current_date = datetime.date.today()

    # Convert the date to a string
    date_string = current_date.strftime("%Y-%m-%d")

    train_df.to_csv(f'models/results/train/{model_name}_{date_string}_noaug.csv', index=False)
    test_df.to_csv(f'models/results/test/{model_name}_{date_string}_noaug.csv', index=False)

    print(model_name, "Done!")
    # TODO: Save results in a csv. 
