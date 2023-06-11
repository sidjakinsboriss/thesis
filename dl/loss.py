import torch
from torch import nn


class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        return (loss * self.weights).mean()
