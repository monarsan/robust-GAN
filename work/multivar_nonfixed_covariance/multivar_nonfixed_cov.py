import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from numpy.lib.function_base import cov
from sys import argv


# nearPD(A) calc projection of A to pd
import numpy as np,numpy.linalg

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return np.array(Yk)



n = int(argv[1])
m = 3*n
eps = float(argv[2])
data_dim = int(argv[3])
mu = float(argv[4])
mu_out = float(argv[5])
par_mu = np.full(data_dim, mu)
par_sd = np.identity(data_dim)
out_mu = np.full(data_dim, mu_out)
true_alpha = [par_mu, par_sd]

exper_iter = int(argv[6])
optim_iter = int(argv[7])
L = int(argv[8])
optim_method = str(argv[9])
learn_par = 1
dicay_par = 0.5

def sigmoid(x):
    return 1/(np.exp(-x) + 1)

def deriv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def g_up(t, s):
    return sigmoid(s) + deriv_sigmoid(s)*(t-s) + (t-s)**2/20

def g_lo(t, s):
    return sigmoid(s) + deriv_sigmoid(s)*(t-s) - (t-s)**2/20



def create_out_cov(data_dim):
    return_cov = np.zeros([data_dim, data_dim])
    for i in range(data_dim):
        for j in range(i+1):
            z = np.random.uniform(0.4, 0.8)
            gamma = np.random.binomial(n = 1,p=0.1)
            return_cov[i][j] = z*gamma
            return_cov[j][i] = z*gamma
    return_cov = return_cov +(np.abs(np.linalg.eig(return_cov)[0]) +0.05)*np.eye(data_dim)
    return return_cov


# 分散を固定しない
out_cov = create_out_cov(data_dim)
res_mean = [0 for i in range(exper_iter)]
res_cov = [0 for i in range(exper_iter)]
for i in tqdm(range(exper_iter)):
    data = np.random.multivariate_normal(mean = par_mu, cov = par_sd, size = int(n*(1-eps)))
    # Gaoの論文の設定
    print("%d/%d" %(i+1, exper_iter))
    contamination = np.random.multivariate_normal(mean = out_mu, cov = out_cov, size = (n - int(n*(1-eps))))
    data = np.concatenate([data, contamination])
    np.random.shuffle(data)
    mean_hist = []
    cov_hist = []
    # 平均は次元ごとにロバスト、分散はロバストでない
    alpha = [np.median(data, axis=0), np.cov(data, rowvar = False)]
    for j in (range(1, optim_iter+1)):
        z = np.random.multivariate_normal(mean=alpha[0], cov=-np.identity(data_dim), size = m)
        def major_func(par, past_par):
            new_beta = par[0:2*data_dim]; new_b = par[2*data_dim]; beta = past_par[0:2*data_dim]; b = past_par[2*data_dim]
            A = np.mean(g_lo(np.dot(np.stack([z, z**2], axis=1).reshape(m, 2*data_dim),new_beta) - new_b, np.dot(np.stack([z, z**2], axis=1).reshape(m, 2*data_dim),beta) - b))
            B = np.mean(g_up(np.dot(np.stack([data, data**2], axis=1).reshape(n, 2*data_dim),new_beta) - new_b, np.dot(np.stack([data, data**2], axis=1).reshape(n, 2*data_dim),beta) - b))
            return -(A-B)

        l = 0; par = np.random.normal(scale = 0.1, size = 2*data_dim + 1)
        while(l<L):
            op = minimize(major_func, x0 = np.zeros(2*data_dim +1), args = par,  method=optim_method)
            par = op.x
            l+=1
        
        alpha_m = alpha[0]; alpha_v = alpha[1]
        mgrad = (z-alpha_m)
        sigma_grad = z[:,:,np.newaxis] * z[:, np.newaxis, :] - alpha_v
        sig_ = sigmoid(np.dot(np.stack([z, z**2], axis=1).reshape(m, 2*data_dim),par[0:2*data_dim ])- par[2*data_dim])[:,np.newaxis]
        tmp_alpha_m = alpha[0] - learn_par/j**dicay_par * np.mean(mgrad*sig_, axis = 0)
        tmp_alpha_v = alpha[1] - learn_par/j**dicay_par * np.mean(sigma_grad*sig_[:,:,np.newaxis], axis = 0)
        alpha[0], alpha[1] = tmp_alpha_m, tmp_alpha_v
        mean_hist.append(alpha[0])
        cov_hist.append(alpha[1])
    res_mean[i] = mean_hist
    res_cov[i] = cov_hist


save_dir_name = str(argv[10])
file_name = "exp_multivar_nonfixed_cov/"+save_dir_name+"/n-"+str(n)+"-eps-"+str(eps*100)+"-p-"+str(data_dim)

np.save(file_name+"mean.npy" ,np.array(res_mean))
np.save(file_name+"cov.npy" ,np.array(res_cov))
np.save(file_name+"out_cov.npy" ,np.array(out_cov))
print("file saved to "+ file_name)