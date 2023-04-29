import torch
import torch.nn as nn


class Discriminator_mu(nn.Module):
    def __init__(self, data_dim) -> None:
        self.data_dim = data_dim
        self.parameters = nn.Parameter(torch.zoros(self.data_dim))
    
    def forward(self, x):
        return torch.sum(x * self.parameters, dim=1)
    
    
class Generator_mu(nn.Module):
    def __init__(self, data_dim) -> None:
        self.data_dim = data_dim
        self.mu = nn.parameter(torch.zeros(self.data_dim))
        
    def forward(self, x):
        return self.mu + x
    