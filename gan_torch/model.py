import torch
import torch.nn as nn


#  models for estimating mean
class Discriminator_linear(nn.Module):
    def __init__(self, data_dim) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.params = nn.Parameter(torch.randn(self.data_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return torch.sum(x * self.params, dim=1) - self.bias
    

class Discriminator_quadraric(nn.Module):
    def __init__(self, data_dim) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.par_ord1 = nn.Parameter(torch.randn(self.data_dim) * 0.1)
        self.par_ord2 = nn.Parameter(torch.randn(self.data_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        ord1 = torch.sum(x * self.par_ord1, dim=1)
        x2 = x ** 2
        ord2 = torch.sum(x2 * self.par_ord2, dim=1)
        return ord1 + ord2 - self.bias
    
    
class Generator_mu(nn.Module):
    def __init__(self, data_dim) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.est_mean = nn.Parameter(torch.zeros(self.data_dim))
        
    def forward(self, x):
        return self.est_mean + x
    
    
# models for estimating covariance  
class Discriminator_sigma(nn.Module):
    def __init__(self, data_dim) -> None:
        super().__init__()
        self.data_dim = data_dim
        matrix = torch.randn(self.data_dim, self.data_dim) * 0.1
        matrix = 0.5 * (matrix + matrix.T)
        self.params = nn.Parameter(matrix)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # todo : check if this is correct
        Ax = torch.matmul(self.params, x.T)
        x = torch.dot(Ax, x)
        return x - self.bias
    
    
class Generator_sigma(nn.Module):
    def __init__(self, data_dim) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.params = nn.Parameter(torch.randn(self.data_dim, self.data_dim))

    def forward(self, x):
        return torch.matmul(self.params, x.T).T
    