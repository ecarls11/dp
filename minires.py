from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class MiniBlock(nn.Module):
    def __init__(self, in_channels=10, num_channels=10, out_channels=10, kernel_size=5, stride=1):
        super(MiniBlock, self).__init__()
        # might or might not need stride
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)/2)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)/2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class MiniRes(nn.Module):
    def __init__(self):
        super(MiniRes, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2)
        self.block1 = MiniBlock(16, 16, 16, kernel_size=3, stride=1)
        self.block2 = MiniBlock(16, 16, 32, kernel_size=3, stride=1)
        self.block3 = MiniBlock(32, 32, 64, kernel_size=3, stride=2)
        self.block4 = MiniBlock(64, 64, 64, kernel_size=3, stride=2) # output is 64x8x8
        self.fc1 = nn.Linear(1024, 1000)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.avg_pool2d(x, kernel_size=2)
        #x = x.view(-1, 1024)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
