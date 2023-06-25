import torch
import numpy as np
import matplotlib.pyplot as plt
from .data import contaminated_data
from .model import Discriminator_linear, Discriminator_quadraric, Generator_mu
from torch.utils.data import DataLoader
from tqdm import trange


class gan():
    def __init__(self, data_dim: int, eps: float, device=None) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.eps = eps
        self.device = device if device is not None \
                      else torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    def dist_init(self, true_mean, out_mean, true_cov, out_cov):
        self.true_mean = torch.tensor(true_mean, dtype=torch.float32)
        self.out_mean = torch.tensor(out_mean, dtype=torch.float32)
        if true_cov is not None:
            self.true_cov = torch.tensor(true_cov, dtype=torch.float32)
        if out_cov is not None:
            self.out_cov = torch.tensor(out_cov, dtype=torch.float32)
            
    def data_init(self, data_size, batch_size):
        self.data = contaminated_data(data_size,
                                      self.true_mean,
                                      self.out_mean,
                                      self.true_cov,
                                      self.out_cov,
                                      self.eps)
        self.true_data = self.data.true_sumple
        self.out_data = self.data.out_sumple
        self.batch_size = batch_size
        if self.device != 'cpu':
            self.dataloader = DataLoader(self.data,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=4)
        else:
            self.dataloader = DataLoader(self.data,
                                         batch_size=batch_size,
                                         shuffle=True)



                

        
        
        
        
        
        
        
        
        
