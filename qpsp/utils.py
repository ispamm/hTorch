import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_RGB(input):
    img = torch.zeros(tuple([input.shape[1]]) + tuple([input.shape[2]]) + tuple([3]))
    img[:, :, 0] = input[4]
    img[:, :, 1] = input[2]
    img[:, :, 2] = input[1]
    plt.imshow((img * 255).detach().numpy().astype(np.uint8))

def classes_statistics(inputs, targets):
  
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    
    targets = targets.view(-1)
    
    tp = (inputs * targets).sum()
    fp = ((1 - targets) * inputs).sum()
    fn = (targets * (1 - inputs)).sum()
    tn = ((1 - targets) * (1 - inputs)).sum()

    return tp, fp, fn, tn

def f1_score(inputs, targets, smooth=1):
    tp, fp, fn, tn = classes_statistics(inputs, targets)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1