from math import log

import numpy as np
import torch
import torch.nn as nn


class WeightedMultilabel(nn.Module):
    def __init__(self, weights):
        super(WeightedMultilabel, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, logits, labels):
        loss = self.criterion(logits, labels)
        res = [(self.weights * sample_loss).mean() for sample_loss in loss]

        res = torch.mean(torch.stack(res, dim=0), dim=0)

        return res


class WBCEWithLogitLoss(nn.Module):
    def __init__(self, class_weights):
        super(WBCEWithLogitLoss, self).__init__()
        self.wp = class_weights['positive_weights']
        self.wn = class_weights['negative_weights']

    def forward(self, logits, targets):
        loss = float(0)
        eps = 1e-10

        res = []

        for k in range(len(logits)):
            target = targets[k]
            logit = logits[k]

            for i, key in enumerate(self.wp.keys()):
                first_term = self.wp[key] * target[i].item() * log(logit[i].item() + eps)
                second_term = self.wn[key] * (1 - target[i].item()) * log(1 - logit[i].item() + eps)
                loss -= first_term + second_term

            res.append(loss)

        res = torch.tensor(res)

        return torch.mean(res)
