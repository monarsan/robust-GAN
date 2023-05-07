import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Discriminator_linear, Discriminator_quadraric, Generator_mu
from tqdm import trange
from gan_torch import gan


class Mu(gan):
    def __init__(self, data_dim, eps, device) -> None:
        super().__init__(data_dim, eps, device)
        
    def dist_init(self, true_mean, out_mean, true_cov=None, out_cov=None):
        super().dist_init(true_mean, out_mean, true_cov, out_cov)
        self.true_cov = torch.eye(self.data_dim, dtype=torch.float32)
        self.out_cov = torch.eye(self.data_dim, dtype=torch.float32)
    
    def data_init(self, data_size, batch_size):
        super().data_init(data_size, batch_size)
        self.trru_mean = self.true_mean.to(self.device)
        
    def model_init(self, D_model='quadratic'):
        assert D_model in ['quadratic', 'linear']
        if D_model == 'quadratic':
            self.D = Discriminator_quadraric(self.data_dim).to(self.device)
        elif D_model == 'linear':
            self.D = Discriminator_linear(self.data_dim).to(self.device)
        self.G = Generator_mu(self.data_dim).to(self.device)
        self.data_median = torch.median(self.data.data, dim=0)[0].to(self.device)
        self.G.est_mean.data = self.data_median
        
    def optimizer_init(self, lr_d, lr_g, d_steps, g_steps, weight_decay_d=0, weight_decay_g=0):
        self.D_optimizer = torch.optim.SGD(self.D.parameters(), lr=lr_d)
        self.G_optimizer = torch.optim.SGD(self.G.parameters(), lr=lr_g)
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.weight_decay_d = weight_decay_d
        self.weight_decay_g = weight_decay_g
    
    def fit(self, epochs):
        self.loss_D = []
        self.loss_G = []
        self.mean_err_record = []
        self.mean_est_record = []
        z_b = torch.zeros(self.batch_size, self.data_dim).to(self.device)
        current_d_step = 1
        for ep in trange(epochs):
            loss_D_ep = []
            loss_G_ep = []
            for _, data in enumerate(self.dataloader):
                # train D
                self.D.train()
                self.D.zero_grad()
                #  D loss
                x_real = data.to(self.device)
                mean_in = x_real.mean(dim=0).detach()
                std_in = x_real.std(dim=0).detach()
                x_real_normalized = (x_real - mean_in) / std_in
                d_real_score = self.D(x_real_normalized)
                mean_out = d_real_score.mean().detach()
                std_out = d_real_score.std().detach()
                d_real_score_normalized = (d_real_score - mean_out) / std_out
                d_real_loss = torch.sigmoid(d_real_score_normalized)
                #  G loss
                x_fake = self.G(z_b.normal_())
                x_fake_normalized = (x_fake - mean_in) / std_in
                d_fake_score = self.D(x_fake)
                d_fake_score_normalized = (d_fake_score - mean_out) / std_out
                d_fake_loss = - torch.sigmoid(d_fake_score)
                reg_d = self.weight_decay_d * self.D.norm()
                d_loss = d_fake_loss + d_real_loss
                d_loss = d_loss.mean() + reg_d
                d_loss.backward()
                loss_D_ep.append(d_loss.item())
                self.D_optimizer.step()
                if current_d_step < self.d_steps:
                    current_d_step += 1
                    continue
                else:
                    current_d_step = 1
                
                #  train G
                self.D.eval()
                for _ in range(self.g_steps):
                    self.G.zero_grad()
                    x_fake = self.G(z_b.normal_())
                    x_fake_normalized = (x_fake - mean_in) / std_in
                    d_fake_score = self.D(x_fake_normalized)
                    d_fake_score_normalized = (d_fake_score - mean_out) / std_out
                    g_loss = torch.sigmoid(d_fake_score_normalized).mean()
                    reg = self.weight_decay_g * (self.G.est_mean - self.data_median).norm(p=2)**2
                    g_loss = g_loss + reg
                    g_loss.backward()
                    loss_G_ep.append(g_loss.item())
                    self.G_optimizer.step()
            self.mean_err_record.append(
                (self.G.est_mean.data - self.true_mean).norm(p=2).item()
            )
            self.mean_est_record.append(self.G.est_mean.data.numpy())
            self.loss_D.append(np.mean(loss_D_ep))
            self.loss_G.append(np.mean(loss_G_ep))
        self.mean_err_record = np.array(self.mean_err_record)
    
    def plot(self):
        col = 4
        row = 1
        plt.figure(figsize=(25, 5))
        plt.subplot(row, col, 1)
        plt.plot(self.loss_D)
        plt.title('loss_D')
        plt.subplot(row, col, 2)
        plt.plot(self.loss_G)
        plt.title('loss_G')
        plt.subplot(row, col, 3)
        plt.plot(self.mean_err_record)
        plt.title('Error')
        plt.subplot(row, col, 4)
        plt.plot(self.mean_est_record)
        plt.title('Mean est record')
        plt.show()
