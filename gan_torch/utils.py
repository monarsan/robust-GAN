import numpy as np
from scipy.stats import kendalltau, t, norm


def kendall(X):
    # X is torch.tensor N by p
    # scaling factor
    medX = np.median(X, axis=0)
    X = X - medX
    # median absolute deviation
    s = np.median(np.abs(X), axis=0)
    # std = k * MAD with k = 1/F^{-1}(3/4), where F is dist of real
    k = 1 / norm.ppf(3 / 4)
    s = k * s
    # sub-sampling
    _, p = X.shape
    corr = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1):
            corr[i, j] = np.sin(np.pi / 2 * kendalltau(X[:, i], X[:, j])[0])
            corr[j, i] = corr[i, j]
    cov = s.reshape(p, 1) * corr * s.reshape(1, p)
    return cov


def ar_cov(data_dim):
    tmp_cov = np.zeros((data_dim, data_dim))
    for i in range(data_dim):
        for j in range(data_dim):
            tmp_cov[i, j] = 2 ** (- abs(i - j))
    return tmp_cov
