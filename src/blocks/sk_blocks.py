from torch import nn
from attention.sk_branch import SKBranch
from attention.sk_attention import SKAttention

class SKUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, M=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # branches
        self.branches = nn.ModuleList([
            SKBranch(mid_channels, mid_channels, kernel_size=3),
            SKBranch(mid_channels, mid_channels, kernel_size=5, dilation=2)
        ][:M])
        
        self.attention = SKAttention(mid_channels, M=M)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        branch_outs = [branch(out) for branch in self.branches]
        out = self.attention(branch_outs)
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)  # residual
