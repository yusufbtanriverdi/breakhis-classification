import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BalancedFocalLoss(nn.Module):
    """Class for focal loss to overcome imbalance challenge."""
    # Source: https://github.com/c0nn3r/RetinaNet/blob/master/resnet_features.py 

    def __init__(self, focusing_param=2, balance_param=0.25):
        # TODO: Explain focusing and balance params, how to choose them.

        # The gamma parameter is the focusing parameter that specifies 
        # how much higher-confidence correct predictions contribute to the overall loss.

        # If gamma is increased, the rate at which easy-to-classify examples are down-weighted will be higher. 
        # This means that the model will focus more on hard-to-classify examples, which can improve the performance 
        # of the model in cases where there is a class imbalance or when the model is struggling to classify certain
        #  examples.
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):
        # TODO: Why repeat here?
        # cross_entropy = F.cross_entropy(output, target)
        # cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt    = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=1, weight= None, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        
        elif self.reduction == 'sum':
            return focal_loss.sum()
        
        else:
            return focal_loss
