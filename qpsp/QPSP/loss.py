import torch
from .utils import classes_statistics


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.25, beta=0.75, gamma=2):
        tp, fp, fn, _ = classes_statistics(inputs, targets)

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        focal_tversky = (1 - tversky) ** gamma

        return focal_tversky