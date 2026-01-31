from torch import nn
from layers.conv_layers import GroupedConv, DepthwiseConv

class SKBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=32, dilation=1):
        super().__init__()
        padding = (kernel_size - 1)//2 * dilation
        if kernel_size == 3:
            self.conv = DepthwiseConv(in_channels, kernel_size=3, stride=stride, padding=1)
        else:
            self.conv = DepthwiseConv(in_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.relu(out)
