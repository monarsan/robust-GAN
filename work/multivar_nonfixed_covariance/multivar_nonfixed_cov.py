import numpy as np
import numpy.linalg as LA
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from numpy.lib.function_base import cov
from sys import argv


# nearPD(A) calc projection of A to pd
import numpy as np

def _getAplus(A):
    eigval, eigvec = LA.eig(A)
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
out_sd = np.identity(data_dim)
true_alpha = [par_mu, par_sd]

exper_iter = int(argv[6])
optim_iter = int(argv[7])
L = int(argv[8])
optim_method = str(argv[9])
dicay_par = float(argv[10])
par_reg1=float(argv[11])
learn_par = float(argv[12])
init_loc=float(argv[13])

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
    return_cov = return_cov +(np.abs(LA.eig(return_cov)[0]) +0.05)*np.eye(data_dim)
    return return_cov


# 分散を固定しない
out_cov = np.eye(data_dim) #create_out_cov(data_dim)
res_mean = [0 for i in range(exper_iter)]
res_cov = [0 for i in range(exper_iter)]
res_par = [0 for i in range(exper_iter)]
for i in (range(exper_iter)):
    data = np.random.multivariate_normal(mean = par_mu, cov = par_sd, size = int(n*(1-eps)))
    # Gaoの論文の設定
    print("%d/%d" %(i+1, exper_iter))
    contamination = np.random.multivariate_normal(mean = out_mu, cov = out_cov, size = (n - int(n*(1-eps))))
    data = np.concatenate([data, contamination])
    np.random.shuffle(data)
    mean_hist = []
    cov_hist = []
    par_hist = []
    # 平均は次元ごとにロバスト、分散はロバストでない
    alpha = [np.median(data, axis=0), np.cov(data, rowvar = False)]
    par = np.random.normal(loc = 0, scale = 0.1, size = 2*data_dim + 1)
    for j in (range(1, optim_iter+1)):
        z = np.random.multivariate_normal(mean=alpha[0], cov=-np.identity(data_dim), size = m)
        def major_func(par, past_par):
            new_beta = par[0:2*data_dim]; new_b = par[2*data_dim]; beta = past_par[0:2*data_dim]; b = past_par[2*data_dim]
            A = np.mean(g_lo(np.dot(np.stack([z, z**2], axis=1).reshape(m, 2*data_dim),new_beta) - new_b, np.dot(np.stack([z, z**2], axis=1).reshape(m, 2*data_dim),beta) - b))
            B = np.mean(g_up(np.dot(np.stack([data, data**2], axis=1).reshape(n, 2*data_dim),new_beta) - new_b, np.dot(np.stack([data, data**2], axis=1).reshape(n, 2*data_dim),beta) - b))
            reg = LA.norm(new_beta, ord=2)*par_reg1
            return -(A-B - reg)

        l = 0
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
        par_hist.append(par)
    res_mean[i] = mean_hist
    res_cov[i] = cov_hist
    res_par[i] = par_hist


import os
file_name=0
while True:
    path = "result/mean"+str(file_name)+".npy"
    if not os.path.isfile(path):
        break
    file_name+=1


np.save(path,np.array(res_mean))
path = "result/cov"+str(file_name)+".npy"
np.save(path,np.array(res_cov))
path = "result/par"+str(file_name)+".npy"
np.save(path,np.array(res_par))

mean = []; npmu = np.array(res_mean); cov = []; npcov=np.array(res_cov)
for i in range(exper_iter):
    loss_mean = LA.norm(npmu[i]-par_mu, ord = 2, axis=1)
    mean.append(loss_mean[-1])
    
for i in range(exper_iter):
    loss_cov = LA.norm(npcov[i]-np.eye(data_dim), axis=1)
    cov.append(loss_cov[-1])

average_loss_mean = str(np.mean(mean))[:6]
average_loss_cov = str(np.mean(cov))[:6]
std = str(np.std(mean))[:6]

import datetime
date = str(datetime.datetime.now())[:-10][:10]
time = str(datetime.datetime.now())[:-10][10:]

with open("./exper_result.csv", mode="a") as f:
    f.write("\n")
    l =  [file_name, average_loss_mean, average_loss_cov ,n, eps, data_dim,
            mu, mu_out,
            par_reg1, learn_par, dicay_par,
            exper_iter, optim_iter, L, init_loc, std,date, time]
    l = list(map(str, l))
    f.writelines(" ,".join(l))