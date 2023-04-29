import torch
import torch.nn as nn


class Discriminator_mu(nn.Module):
    def __init__(self, data_dim) -> None:
        self.data_dim = data_dim
        self.parameters = nn.Parameter(torch.zoros(self.data_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return nn.Sigmoid(torch.sum(x * self.parameters, dim=1) - self.bias)
    
    
class Generator_mu(nn.Module):
    def __init__(self, data_dim) -> None:
        self.data_dim = data_dim
        self.mu = nn.parameter(torch.zeros(self.data_dim))
        
    def forward(self, x):
        return self.mu + x
    