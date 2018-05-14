from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class MiniBlock(nn.Module):
    def __init__(self, in_channels=10, num_channels=10, out_channels=10, kernel_size=5, stride=1):
        super(MiniBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.stride > 1:
            downsample = nn.Conv2d(residual.size(1), residual.size(1), kernel_size=1, stride=self.stride)
            bn = nn.BatchNorm2d(residual.size(1))
            if torch.cuda.is_available():
                downsample = downsample.cuda()
                bn = bn.cuda()
            residual = downsample(residual)
            residual = bn(residual)
        if self.out_channels != self.in_channels:
            z = Variable(torch.zeros(x.size(0), self.out_channels - self.in_channels, x.size(2), x.size(3)))
            if torch.cuda.is_available():
                z = z.cuda()
            temp = (residual, z)
            residual = torch.cat(temp, 1)
        x += residual
        return F.relu(x)

class MiniRes(nn.Module):
    def __init__(self):
        super(MiniRes, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2)
        self.block1 = MiniBlock(16, 16, 16, kernel_size=3, stride=1)
        self.block2 = MiniBlock(16, 16, 32, kernel_size=3, stride=1)
        self.block3 = MiniBlock(32, 32, 64, kernel_size=3, stride=2) 
        self.block4 = MiniBlock(64, 64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(1024, 1000)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.avg_pool2d(x, kernel_size=2)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
