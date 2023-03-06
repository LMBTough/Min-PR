import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn as nn


class SigNet(nn.Module):
    def __init__(self,kind):
        super(SigNet, self).__init__()
        if kind == 2:
            self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x_sigmoid = self.sigmoid(x)

        return x_sigmoid

