import numpy as np
import numpy.linalg as LA


def create_sparse_cov(data_dim):
    return_cov = np.zeros([data_dim, data_dim])
    for i in range(data_dim):
        for j in range(i + 1):
            z = np.random.uniform(0.4, 0.8)
            gamma = np.random.binomial(n=1, p=0.1)
            return_cov[i][j] = z * gamma
            return_cov[j][i] = z * gamma
    return_cov = return_cov + (np.abs(LA.eig(return_cov)[0].min()) + 0.05) * np.eye(data_dim)
    return return_cov


def create_sigma_ar(data_dim):
    cov = np.zeros([data_dim, data_dim])
    for i in range(data_dim):
        for j in range(data_dim):
            cov[i, j] = 0.5 ** np.abs(i - j)
    return cov


def create_norm_data(data_size, eps, true_mu, true_cov, out_mu, out_cov):
    data_size_true_dist = np.random.binomial(data_size, 1 - eps)
    data_target = np.random.multivariate_normal(mean=true_mu, cov=true_cov, size=data_size_true_dist)
    data_contami = np.random.multivariate_normal(mean=out_mu, cov=out_cov, size=data_size - data_size_true_dist)
    data = np.concatenate([data_target, data_contami], axis=0)
    np.random.shuffle(data)
    return data
