from ast import arg
from genericpath import isfile
import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from numpy.lib.function_base import cov
from sys import argv
from tqdm import tqdm

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


# 分散を単位行列で固定
res = [0 for i in range(exper_iter)]
res_par = [0 for i in range(exper_iter)]
for i in range(exper_iter):
    print("%d/%d" %(i+1, exper_iter))
    data = np.random.multivariate_normal(mean = par_mu, cov = par_sd, size = int(n*(1-eps)))
    contamination = np.random.multivariate_normal(mean = out_mu, cov = out_sd, size = (n - int(n*(1-eps))))
    data = np.concatenate([data, contamination])
    np.random.shuffle(data)
    alpha_hist = []
    par_hist = []
    alpha = [np.median(data, axis=0), np.identity(data_dim)]
    z = np.random.multivariate_normal(mean=alpha[0], cov=alpha[1], size = m)
    par = np.random.normal(loc = 0, scale = 0.1, size = 2*data_dim)
    bias = np.array(np.mean(np.dot(np.stack([z, z**2], axis=1).reshape(m, 2*data_dim),par[0:2*data_dim])))[np.newaxis]
    par = np.concatenate([par, bias], axis = 0)

    for j in tqdm(range(1, optim_iter+1)):
        z = np.random.multivariate_normal(mean=alpha[0], cov=alpha[1], size = m)

        def major_func(par, past_par):
            new_beta = par[0:2*data_dim]; new_b = par[2*data_dim]; beta = past_par[0:2*data_dim]; b = past_par[2*data_dim]
            A = np.mean(g_lo(np.dot(np.stack([z, z**2], axis=1).reshape(m, 2*data_dim),new_beta) - new_b, np.dot(np.stack([z, z**2], axis=1).reshape(m, 2*data_dim),beta) - b))
            B = np.mean(g_up(np.dot(np.stack([data, data**2], axis=1).reshape(n, 2*data_dim),new_beta) - new_b, np.dot(np.stack([data, data**2], axis=1).reshape(n, 2*data_dim),beta) - b))
            reg = np.linalg.norm(new_beta, ord=2)*par_reg1
            return -(A-B -reg)

        l = 0; 
        while(l<L):
            # minimize の初期点0でいいの？
            op = minimize(major_func, x0 = np.zeros(2*data_dim +1), args = par, method=optim_method)
            par = op.x
            l+=1
            
        alpha_m = alpha[0]; alpha_v = alpha[1]
        mgrad = (z-alpha_m)
        sig_ = sigmoid(np.dot(np.stack([z, z**2], axis=1).reshape(m, 2*data_dim),par[0:2*data_dim ])- par[2*data_dim])[:,np.newaxis]
        tmp_alpha = alpha[0] - learn_par/j**dicay_par * np.mean(mgrad*sig_, axis = 0)
        alpha[0] = tmp_alpha
        alpha_hist.append(alpha[0])
        par_hist.append(par)
    res[i] = alpha_hist
    res_par[i] = par_hist

import os
file_name=0
while True:
    path = "result/res"+str(file_name)+".npy"
    if not os.path.isfile(path):
        break
    file_name+=1


np.save(path,np.array(res))
path = "result/par"+str(file_name)+".npy"
np.save(path,np.array(res_par))

mean = []; npmu = np.array(res)
for i in range(exper_iter):
    loss = np.linalg.norm(npmu[i]-par_mu, ord = 2, axis=1)
    mean.append(loss[-1])
average_loss = str(np.mean(mean))[:6]
std = str(np.std(mean))[:6]

import datetime
date = str(datetime.datetime.now())[:-10][:10]
time = str(datetime.datetime.now())[:-10][10:]

with open("./exper_result.csv", mode="a") as f:
    f.write("\n")
    l =  [file_name, average_loss ,n, eps, data_dim,
            mu, mu_out,
            par_reg1, learn_par, dicay_par,
            exper_iter, optim_iter, L, init_loc, std,date, time ]
    l = list(map(str, l))
    f.writelines(" ,".join(l))
