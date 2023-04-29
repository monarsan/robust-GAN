from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.binomial import Binomial
from torch.utils.data import Dataset
from torch import tensor
import torch


class contaminated_data(Dataset):
    '''
    Generate contaminated data
    you can access true sample and outlier sample
    '''
    def __init__(self,
                 data_size: int,
                 true_mean: tensor,
                 out_mean: tensor,
                 true_cov: tensor,
                 out_cov: tensor,
                 eps: float):
        num_true_sample = int(self._binomial(data_size, 1 - eps))
        num_out_sample = data_size - num_true_sample
        true_dist = MultivariateNormal(true_mean, true_cov)
        out_dist = MultivariateNormal(out_mean, out_cov)
        self.true_sumple = true_dist.sample((num_true_sample,))
        self.out_sumple = out_dist.sample((num_out_sample,))
        self.data = torch.cat((self.true_sumple, self.out_sumple))
    
    def _binomial(self, n, p):
        binomial = Binomial(n, p)
        binomial = binomial.sample((1,))
        return binomial[0]
      
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]