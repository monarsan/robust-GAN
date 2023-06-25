import torch
import torch.nn as nn
from .utils import quad_form


#  models for estimating mean
class Discriminator_linear(nn.Module):
    def __init__(self, data_dim) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.params = nn.Parameter(torch.randn(self.data_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return torch.sum(x * self.params, dim=1) - self.bias
    
    def norm(self):
        norm = self.params.norm(p=2) ** 2
        return norm.mean() + self.bias ** 2
    
    
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
    
    def norm(self):
        norm = torch.concat((self.par_ord1, self.par_ord2)).norm(p=2) ** 2
        return norm.mean() + self.bias ** 2
    
    
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
        x = quad_form(x, self.params)
        return x - self.bias
    
    def norm(self):
        A = self.params.norm(p=2) ** 2
        b = self.bias ** 2
        return A + b
    
    def get_par(self):
        return self.params
    
    
class Discriminator_sigma_complex(nn.Module):
    def __init__(self, data_dim) -> None:
        super().__init__()
        self.data_dim = data_dim
        matrix = torch.randn(self.data_dim, self.data_dim) * 0.1
        matrix = 0.5 * (matrix + matrix.T)
        self.params = nn.Parameter(matrix)
        
        matrix = torch.randn(self.data_dim, self.data_dim) * 0.1
        matrix = 0.5 * (matrix + matrix.T)
        self.params2 = nn.Parameter(matrix)
                
        self.param_linear = nn.Parameter(torch.randn(self.data_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        x1 = quad_form(x, self.params)
        x_sq = x ** 2
        x2 = quad_form(x_sq, self.params2)
        x_lin = torch.sum(x * self.param_linear, dim=1)
        return x1 + x2 + x_lin - self.bias
    
    def norm(self):
        A1 = self.params.norm(p='fro') ** 2
        A2 = self.params2.norm(p='fro') ** 2
        lin = self.param_linear.norm(p=2) ** 2
        A = A1 + A2 + lin
        b = self.bias ** 2
        return A + b
    
    def get_par(self):
        return self.params
    
    
class Generator_sigma(nn.Module):
    def __init__(self, data_dim) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.est_sigma = nn.Parameter(torch.randn(self.data_dim, self.data_dim))

    def forward(self, x):
        return torch.matmul(self.est_sigma, x.T).T

    def est_cov(self):
        return torch.matmul(self.est_sigma, self.est_sigma.T)
    