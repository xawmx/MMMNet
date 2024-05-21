import torch
import torch.nn as nn


class FeatureMask(nn.Module):
    def __init__(self, projection_dim):
        super(FeatureMask, self).__init__()
        self.mask = nn.Parameter(torch.zeros(int(projection_dim)))

    def forward(self):
        return torch.sigmoid(torch.ones_like(self.mask) * self.mask)