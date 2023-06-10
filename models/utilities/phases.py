import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import Accuracy, Recall, AveragePrecision, AUROC, MeanAbsolutePercentageError, F1Score, R2Score, CohenKappa
import pandas as pd
import datetime
import numpy as np

def train(model, train_loader, optimizer, criterion, eval_metrics, epoch=-1, cuda=False):

    average_loss = 0
    metric_values = {metric_name: [] for metric_name in eval_metrics.keys()}

    if cuda:
        model.cuda()
        model = nn.DataParallel(model)

    for X, y in train_loader:

        optimizer.zero_grad()

        if cuda:
            X.cuda()
            y.cuda()

        X = Variable(X, requires_grad=True)
        yhat = model(X)

        loss = criterion(yhat, y)

        loss.backward()
        optimizer.step()
        average_loss += loss
        # Compute evaluation metrics
        with torch.no_grad():
             # Convert labels to one-hot vectors and vice-versa.
            y_vectors = torch.from_numpy(np.eye(2)[y].astype(np.int64))
            yhat_labels = np.argmax(yhat, axis=1)
            for metric_name, metric in eval_metrics.items():
                try:
                    # Try as one-hot-vectors.
                    metric_val = metric(yhat_labels, y)
                except ValueError as e:
                    # Try as logits. 
                    metric_val = metric(yhat, y_vectors)
                
                metric_values[metric_name].append(metric_val.item())
        
    average_loss /= len(train_loader),
    epoch_scores = {
        'Epoch': epoch + 1,
        'Average Loss': average_loss.detach().numpy()
        **{metric_name: sum(metric_values[metric_name]) / len(metric_values[metric_name]) for metric_name in eval_metrics}
    }

    print(epoch_scores)
    return epoch_scores

def test(model, test_loader, criterion, eval_metrics, epoch=-1, cuda=False, mode='binary'):

    average_loss = 0
    metric_values = {metric_name: [] for metric_name in eval_metrics.keys()}

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

            # Convert labels to one-hot vectors and vice-versa.
            y_vectors = torch.from_numpy(np.eye(2)[y])
            yhat_labels = np.argmax(yhat, axis=1)
            for metric_name, metric in eval_metrics.items():
                try:
                    # Try as one-hot-vectors.
                    metric_val = metric(yhat_labels, y)
                except ValueError as e:
                    # Try as logits. 
                    metric_val = metric(yhat, y_vectors)

                metric_values[metric_name].append(metric_val.item())
        
    average_loss /= len(test_loader),
    epoch_scores = {
        'Epoch': epoch + 1,
        'Average Loss': average_loss,
        **{metric_name: sum(metric_values[metric_name]) / len(metric_values[metric_name]) for metric_name in eval_metrics}
    }

    print(epoch_scores)
    return epoch_scores

def eval(model, test_loader, train_loader, optimizer, criterion, num_epochs= 50, mode='binary'):

    eval_metrics = {
    'accuracy_score': Accuracy(task=mode),
    'roc_auc_score': AUROC(task=mode),
    'average_precision_score' : AveragePrecision(task=mode),
    'mean_absolute_percentage_error' : MeanAbsolutePercentageError(),
    'f1_score' : F1Score(mode),
    'r2_score' : R2Score(),
    'recall_Score' : Recall(mode),
    'cohen_kappa_score' : CohenKappa(mode)
    }
    # TODO:  Add pattern recognition rate.

    for t in tqdm(range(num_epochs), desc='Training on Breast Histopathology Dataset', unit='epoch'):
        print(f"Epoch {t+1}\n-------------------------------")
        train_scores = train(model, train_loader, optimizer, criterion, eval_metrics, epoch = t,)
        test_scores = test(model, test_loader, criterion, eval_metrics, epoch= t)
        if t == 0:
            train_df = pd.DataFrame(train_scores)
            test_df = pd.DataFrame(test_scores)
        else:
            train_df = pd.concat([train_df, pd.DataFrame(train_scores)])
            test_df = pd.concat([test_df, pd.DataFrame(test_scores)])
    
    # Save the DataFrame as a CSV file
    model_name = model.__class__.__name__

    # Get the current date
    current_date = datetime.date.today()

    # Convert the date to a string
    date_string = current_date.strftime("%Y-%m-%d")

    train_df.to_csv(f'models/results/train/{model_name}_{date_string}.csv', index=False)
    test_df.to_csv(f'models/results/test/{model_name}_{date_string}.csv', index=False)

    print(model_name, "Done!")
    # TODO: Save results in a csv. 
