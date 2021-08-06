import torch
import torch.nn as nn

def classes_statistics(inputs, targets):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    tp = (inputs * targets).sum()
    fp = ((1 - targets) * inputs).sum()
    fn = (targets * (1 - inputs)).sum()
    tn = ((1 - targets) * (1 - inputs)).sum()

    return tp, fp, fn, tn

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.25, beta=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        tp, fp, fn, _ = classes_statistics(inputs, targets)

        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky