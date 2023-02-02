import numpy as np
import numpy.linalg as LA


def sigmoid(x):
    return 1/(np.exp(-x) + 1)

def deriv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def g_up(t, s):
    return sigmoid(s) + deriv_sigmoid(s)*(t-s) + (t-s)**2/20

def g_lo(t, s):
    return sigmoid(s) + deriv_sigmoid(s)*(t-s) - (t-s)**2/20

#numpy opration
def sample_wise_vec_mat_vec(A, x):
    r = (x[:,np.newaxis,:]@A@x[:,:,np.newaxis])
    return np.squeeze(r)

def sample_wise_outer_product(x, y):
    return x[:,:,np.newaxis]@y[:,np.newaxis,:]

def mat_vec(A, x):
    return A@x

def vec_mat_vec(A, x):
    return x@A@x


def mean_outer_product(z1, z2): #(n, d)*(n, d)  vector_wise_dot->  (n, d, d)  mean-> (d, d)
    outer_product = z2[:, np.newaxis, :]*z1[:, :, np.newaxis]
    return np.mean(outer_product, axis = 0)

def sample_wise_mat_vec(A, x):
    return x@A


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
    k = 1/norm.ppf(3/4)
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

def init_covariance(data, method='mle'):
    if method=='kendall':
        cov = kendall(data)
        u, s, vt = np.linalg.svd(cov)
        cov=np.matmul(np.diag(s)**(1/2), vt).T

    elif method=='mle':
        cov = np.cov(data, rowvar=False)
        cov = LA.cholesky(cov)
    else :
        raise ValueError('method is mle or kendall !')

    return cov


def init_discriminator(data, method='mle'):
    _, d = data.shape
    if method =='random':
        cov = np.random.normal(0, 0.1, d**2).reshape(d,d)
        cov = (cov+cov.T)/2
    elif method=='kendall':
        cov = kendall(data)
        u, s, vt = np.linalg.svd(cov)
        cov=np.matmul(np.diag(s)**(1/2), vt).T

    elif method=='mle':
        cov = np.cov(data, rowvar=False)
        cov = LA.cholesky(cov)
    
    else :
        raise ValueError('method is mle or kendall !')

    return (cov@cov.T).reshape(d**2)