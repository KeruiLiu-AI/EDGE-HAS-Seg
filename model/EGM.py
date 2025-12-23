import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def get_sobel_3d(in_chan, out_chan):
    filter_x = np.array([
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    ]).astype(np.float32)
    filter_y = np.array([
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    ]).astype(np.float32)
    filter_z = np.array([
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_z = filter_z.reshape((1, 1, 3, 3, 3))
    filter_z = np.repeat(filter_z, in_chan, axis=1)
    filter_z = np.repeat(filter_z, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_z = torch.from_numpy(filter_z)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    filter_z = nn.Parameter(filter_z, requires_grad=False)
    conv_x = nn.Conv3d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv3d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    conv_z = nn.Conv3d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_z.weight = filter_z
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm3d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm3d(out_chan))
    sobel_z = nn.Sequential(conv_z, nn.BatchNorm3d(out_chan))

    return sobel_x, sobel_y, sobel_z

def run_sobel_3d(conv_x, conv_y, conv_z, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g_z = conv_z(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + torch.pow(g_z, 2))
    return torch.sigmoid(g) * input

class EGAB(nn.Module):
    def __init__(self, in_channels):
        super(EGAB, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(in_channels)
        self.ban = nn.BatchNorm3d(in_channels)
        self.sobel_x1, self.sobel_y1, self.sobel_z1 = get_sobel_3d(in_channels, 1)

    def forward(self, x):
        y = run_sobel_3d(self.sobel_x1, self.sobel_y1, self.sobel_z1, x)
        y = F.relu(self.bn(y))
        y = self.conv1(y)
        y = x + y
        y = self.conv2(y)
        y = F.relu(self.ban(y))

        return y