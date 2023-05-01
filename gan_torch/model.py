import torch
import torch.nn as nn


class Discriminator_mu(nn.Module):
    def __init__(self, data_dim) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.params = nn.Parameter(torch.randn(self.data_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return torch.sum(x * self.params, dim=1) - self.bias
    
    
class Generator_mu(nn.Module):
    def __init__(self, data_dim) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.est_mean = nn.Parameter(torch.zeros(self.data_dim))
        
    def forward(self, x):
        return self.est_mean + x
    