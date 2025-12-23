import torch
import torch.nn as nn
from torch.nn import functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction = 16):
        super(ChannelAttention3D, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        y = F.adaptive_avg_pool3d(x, 1).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1, 1)
        return x * y


class MultiChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiChannelSpatialAttention, self).__init__()
        self.conv = nn.Conv3d(in_channels = 2, out_channels = 1, kernel_size = 7, padding = 3, bias = False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(x_cat))
        return x * attention