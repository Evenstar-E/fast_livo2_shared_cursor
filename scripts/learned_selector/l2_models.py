"""Fixed CPU TCN: no temporal/batch normalization and no right padding."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv(nn.Module):
    def __init__(self, channels, dilation, kernel):
        super().__init__()
        self.left = dilation*(kernel-1)
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0)))


class Block(nn.Module):
    def __init__(self, channels, dilation, kernel):
        super().__init__()
        self.a = CausalConv(channels, dilation, kernel)
        self.b = CausalConv(channels, dilation, kernel)

    def forward(self, x):
        return F.relu(x+self.b(F.relu(self.a(x))))


class SelectorTCN(nn.Module):
    def __init__(self, config, features=22, length=64):
        super().__init__()
        c = config['channels']
        self.input = nn.Conv1d(features*2+1, c, 1)
        self.blocks = nn.Sequential(*(Block(c,d,config['kernel']) for d in config['dilations']))
        self.fault = nn.Conv1d(c, 1, 1)
        self.onset = nn.Conv1d(c, length+2, 1)

    def forward(self, x):
        h = self.blocks(self.input(x))
        return self.fault(h)[:, 0], self.onset(h)
