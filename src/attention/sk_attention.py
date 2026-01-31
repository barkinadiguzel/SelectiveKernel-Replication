import torch
from torch import nn
import torch.nn.functional as F
from attention.gap_pool import GAP

class SKAttention(nn.Module):
    def __init__(self, channels, M=2, reduction=16):
        super().__init__()
        self.M = M  
        self.channels = channels
        self.gap = GAP()
        d = max(channels // reduction, 32)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True)
        )

        self.fcs = nn.ModuleList([nn.Linear(d, channels, bias=False) for _ in range(M)])
    
    def forward(self, features): 
        U = sum(features)  # Fuse
        s = self.gap(U)    # GAP
        z = self.fc(s)     
        
        a = []
        for fc in self.fcs:
            a.append(fc(z).unsqueeze(1)) 
        
        a = torch.cat(a, dim=1)  
        a = F.softmax(a, dim=1)  
        
        out = 0
        for i in range(self.M):
            out += features[i] * a[:,i,:,:,:]  
        return out
