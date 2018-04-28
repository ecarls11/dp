from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class MiniG(nn.Module):
    def __init__(self):
        super(MiniG, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(12544, 1024)
        self.bnfc1 = nn.BatchNorm2d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bnfc2 = nn.BatchNorm2d(1024)
        self.fc3 = nn.Linear(1024, 200)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.dropout(x, training=self.training)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.dropout(x, training=self.training)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(x)
        x = F.dropout(x, training=self.training)

        x = x.view(-1, 12544)
        x = F.relu(self.fc1(x))
        x = self.bnfc1(x)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.bnfc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
