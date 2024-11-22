import torch
import torch.nn as nn
from torch import sqrt

def normc_fn(m):#初始化
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ResidualBlock(nn.Module):#残差网络
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),#随机丢弃神经元
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        if in_features != out_features:
            self.residual = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        return self.block(x) + self.residual(x)


# The base class for an actor. Includes functions for normalizing state (optional)
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.env_name = None

  def initialize_parameters(self):
    self.apply(normc_fn)




