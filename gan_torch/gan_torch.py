import torch
import numpy as np
import matplotlib.pyplot as plt
from data import contaminated_data
from model import Discriminator_mu, Generator_mu
from torch.utils.data import DataLoader


class gan():
    def __init__(self, data_dim: int, eps: float, device=None) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.eps = eps
         self.device = device if device is not None \
                      else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
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
        self._out_data = self.data.out_sumple
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)

class mu(gan):
    def __init__(self, data_dim, eps, device) -> None:
        super().__init__(data_dim, eps, device)
        
    def dist_init(self, true_mean, out_mean, true_cov=None, out_cov=None):
        super().dist_init(true_mean, out_mean, true_cov, out_cov)
        self.true_cov = torch.tensor(true_cov, dtype=torch.float32)
        self.out_cov = torch.tensor(out_cov, dtype=torch.float32)
    
    def data_init(self, data_size, batch_size):
        super().dist_init(data_size, batch_size)
        
        
    def model_init(self):
        self.D = Discriminator_mu(self.data_dim).to(self.device)
        self.G = Generator_mu(self.data_dim).to(self.device)
        data_median = torch.median(self.data.data, dim=0)
        self.G.data = data_median
        
    def optimizer_init(self, lr_d, lr_g, d_steps, g_steps):
        self.D_optimizer = torch.optim.SGD(self.D.parameters(), lr=lr_d)
        self.G_optimizer = torch.optim.SGD(self.G.parameters(), lr=lr_g)
        self.d_steps = d_steps
        self.g_steps = g_steps
    
    def fit(self, epochs):
        self.loss_D = [] 
        self.loss_G = []
        self.mean_err_record = []
        self.mean_est_record = []
        z_b = torch.zeros(self.batch_size, self.p).to(self.device)
        one_b = torch.ones(self.batch_size).to(self.device)
        criterion = torch.nn.BCELoss()
        current_d_step = 1
        for ep in range(epochs):
            loss_D_ep = []
            loss_G_ep = []
            for _, data in enumerate(self.dataloader):
                # train D
                self.D.train()
                self.D.zero_grad()
                # D loss
                x_real = data.to(self.device)
                d_real_score = self.D(x_real)
                d_real_loss = d_real_score.mean()
                d_real_loss = criterion(d_real_loss, z_b)
                # G loss
                x_fake = self.G(z_b)
                d_fake_score = self.D(x_fake)
                d_fake_score = d_fake_score.mean()
                d_fake_loss = criterion(d_fake_score, one_b)
                d_loss = d_fake_loss - d_real_loss
                d_loss.backward()
                loss_D_ep.append(d_loss.item())

                
class sigma(gan):
    def __init__(self, data_dim, eps, device) -> None:
        super().__init__(data_dim, eps, device)
        
    def dist_init(self, true_mean, out_mean, true_cov, out_cov):
        super().dist_init(true_mean, out_mean, true_cov, out_cov)
        self.true_mean = torch.tensor(true_mean, dtype=torch.float32)
        
    def data_init(self, data_size):
        super().data_init(data_size)
        
        
        
        
        
        
        
        
        
