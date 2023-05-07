import torch
import numpy as np
import matplotlib.pyplot as plt
from data import contaminated_data
from model import Discriminator_sigma, Generator_sigma
from torch.utils.data import DataLoader
from tqdm import trange
from gan_torch import gan
from utils import kendall
from torch.optim.lr_scheduler import StepLR


class Sigma(gan):
    def __init__(self, data_dim, eps, device) -> None:
        super().__init__(data_dim, eps, device)
        
    def dist_init(self, true_mean, out_mean, true_cov, out_cov):
        super().dist_init(true_mean, out_mean, true_cov, out_cov)
        self.true_mean = torch.tensor(true_mean, dtype=torch.float32)
        
    def data_init(self, data_size, batch_size):
        super().data_init(data_size, batch_size)
        self.true_cov = torch.tensor(self.true_cov, dtype=torch.float32).to(self.device)
    
    def model_init(self):
        self.D = Discriminator_sigma(self.data_dim).to(self.device)
        self.G = Generator_sigma(self.data_dim).to(self.device)
        self.G_init = torch.tensor(kendall(self.data.data.numpy())).to(self.device)
        self.G.est_sigma.data = torch.tensor(self.G_init, dtype=torch.float32).to(self.device)
        
    def optimizer_init(self, lr_d, lr_g, d_steps, g_steps,
                       weight_decay_d=0, weight_decay_g=0,
                       step_size=200, gamma=0.2):
        self.D_optimizer = torch.optim.SGD(self.D.parameters(), lr=lr_d)
        self.G_optimizer = torch.optim.SGD(self.G.parameters(), lr=lr_g)
        self.scheduler = StepLR(self.G_optimizer, step_size=step_size, gamma=gamma)
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.weight_decay_d = weight_decay_d
        self.weight_decay_g = weight_decay_g
    
    def fit(self, epochs):
        self.loss_D = []
        self.loss_G = []
        self.sigma_err_record = []
        self.sigma_est_record = []
        z_b = torch.zeros(self.batch_size * 3, self.data_dim).to(self.device)
        current_d_step = 1
        for ep in trange(epochs):
            loss_D_ep = []
            loss_G_ep = []
            for _, data in enumerate(self.dataloader):
                # train D
                self.D.train()
                self.D.zero_grad()
                # D loss
                x_real = data.to(self.device)
                mean_in = x_real.mean(dim=0).detach()
                std_in = torch.std(x_real, dim=0).detach() + 1e-6
                x_real_normalized = (x_real - mean_in)/std_in
                d_real_score = self.D(x_real_normalized)
                mean_out = d_real_score.mean(dim=0).detach()
                std_out = torch.std(d_real_score, dim=0).detach() + 1e-6
                d_real_score_normalized = (d_real_score - mean_out)/std_out
                d_real_score = torch.sigmoid(d_real_score_normalized)
                # G loss
                x_fake = self.G(z_b.normal_())
                x_fake_normalized = (x_fake - mean_in)/std_in
                d_fake_score = self.D(x_fake_normalized)
                d_fake_score_normalized = (d_fake_score - mean_out)/std_out
                d_fake_score = - torch.sigmoid(d_fake_score_normalized)
                d_loss = d_fake_score + d_real_score
                reg_d = self.weight_decay_d * self.D.norm()
                d_loss = d_loss.mean() + reg_d
                d_loss.backward()
                loss_D_ep.append(d_loss.item())
                self.D_optimizer.step()
                if current_d_step < self.d_steps:
                    current_d_step += 1
                    continue
                else:
                    current_d_step = 1
                
                # train G
                self.D.eval()
                for _ in range(self.g_steps):
                    self.G.zero_grad()
                    x_fake = self.G(z_b.normal_())
                    x_fake_normalized = (x_fake - mean_in)/std_in
                    d_fake_score = self.D(x_fake_normalized)
                    d_fake_score_normalized = (d_fake_score - mean_out)/std_out
                    g_loss = torch.sigmoid(d_fake_score_normalized).mean()
                    reg = self.weight_decay_g * (self.G.est_sigma - self.G_init).norm(p=2) ** 2
                    g_loss = g_loss + reg
                    g_loss.backward()
                    loss_G_ep.append(g_loss.item())
                    self.G_optimizer.step()
            self.scheduler.step()
            self.sigma_err_record.append(
                (self.G.est_cov() - self.true_cov).norm(p=2).item()
            )
            self.sigma_est_record.append(self.G.est_cov().cpu().detach().numpy())
            self.loss_D.append(np.mean(loss_D_ep))
            self.loss_G.append(np.mean(loss_G_ep))
        self.sigma_err_record = np.array(self.sigma_err_record)
        self.sigma_est_record = np.array(self.sigma_est_record)
            
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
        plt.plot(self.sigma_err_record)
        plt.title('Error')
        plt.subplot(row, col, 4)
        plt.plot(self.sigma_est_record.reshape(-1, self.data_dim ** 2))
        plt.title('Sigma est record')
        plt.show()
            