import torch
import torch.nn as nn

class Multiplier(nn.Module):
  def __init__(self,size=1,no_bias=False):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(size))
    self.bias = nn.Parameter(torch.zeros(size))
    self.no_bias = no_bias

  def forward(self, x):
    if self.no_bias:
      return x * self.weight
    else:
      return x * self.weight + self.bias/100