import torch
import numpy as np
import matplotlib.pyplot as plt
from data import contaminated_data
from model import Discriminator_mu, Generator_mu


class gan():
    def __init__(self, data_dim: int, eps: float) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.eps = eps
    
    def dist_init(self, true_mean, out_mean, true_cov, out_cov):
        self.true_mean = torch.tensor(true_mean, dtype=torch.float32)
        self.out_mean = torch.tensor(out_mean, dtype=torch.float32)
        if true_cov is not None:
            self.true_cov = torch.tensor(true_cov, dtype=torch.float32)
        if out_cov is not None:
            self.out_cov = torch.tensor(out_cov, dtype=torch.float32)
            
    def data_init(self, data_size):
        self.data = contaminated_data(data_size,
                                      self.true_mean,
                                      self.out_mean,
                                      self.true_cov,
                                      self.out_cov,
                                      self.eps)
        self.true_data = self.data.true_sumple
        self._out_data = self.data.out_sumple


class mu(gan):
    def __init__(self, data_dim, eps) -> None:
        super().__init__(data_dim, eps)
        
    def dist_init(self, true_mean, out_mean, true_cov=None, out_cov=None):
        super().dist_init(true_mean, out_mean, true_cov, out_cov)
        self.true_cov = torch.tensor(true_cov, dtype=torch.float32)
        self.out_cov = torch.tensor(out_cov, dtype=torch.float32)
    
    def data_init(self, data_size):
        super().dist_init(data_size)
       
        
class sigma(gan):
    def __init__(self, data_dim, eps) -> None:
        super().__init__(data_dim, eps)
        
    def dist_init(self, true_mean, out_mean, true_cov, out_cov):
        super().dist_init(true_mean, out_mean, true_cov, out_cov)
        self.true_mean = torch.tensor(true_mean, dtype=torch.float32)
        
    def data_init(self, data_size):
        super().data_init(data_size)
        
        
        