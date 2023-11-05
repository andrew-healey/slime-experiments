import torch
import torch.nn as nn

class Multiplier(nn.Module):
  def __init__(self,size=1):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(size))
    self.bias = nn.Parameter(torch.zeros(size))

  def forward(self, x):
    return x * self.weight + self.bias