import torch
from torch import nn

class GAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        return self.pool(x).view(x.size(0), -1) 
