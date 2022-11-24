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
import sys
sys.path.append("../")
from libs.create import create_norm_data, create_out_cov
from libs.functions import sigmoid, g_lo, g_up, mean_outer_product, deriv_sigmoid
import numpy.linalg as LA

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
lr_u = float(argv[14])

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
    #par = np.random.normal(loc = 0, scale = 0.1, size = 2*data_dim + 1)
    par = np.random.normal(loc = 0, scale = 0.1, size = 2*data_dim)
    bias = np.array(np.mean(np.dot(np.stack([z**2, z], axis=1).reshape(m, 2*data_dim),par[0:2*data_dim])))[np.newaxis]
    par = np.concatenate([par, bias], axis = 0)
    for j in tqdm(range(1, optim_iter+1)):
        z = np.random.multivariate_normal(mean=alpha[0], cov=alpha[1], size = m)
        l = 0
        z_sq = z**2
        data_sq = data**2
        while l<L:
            par_a = par[0:data_dim]; par_b = par[data_dim: 2*data_dim]; bias = par[-1]
            t0_z = z_sq@par_a + z@par_b - bias #shape (m,)
            t0_data = data_sq@par_a + data@par_b - bias # (n,)
           
            #連立方程式の行列を求める Ax = b
            # ここからがMMアルゴリズムの計算
            A1 = -1/10 * (mean_outer_product(z_sq, z_sq) + mean_outer_product(data_sq, data_sq))
            A2 = -1/10 * (mean_outer_product(z_sq, z)    + mean_outer_product(data_sq, data))
            A3 =  1/10 * (z_sq.mean(axis=0)                   +data_sq.mean(axis=0))
            b1 = - ((deriv_sigmoid(t0_z)+t0_z/10)[:, np.newaxis]*z_sq).mean(axis=0) + ((deriv_sigmoid(t0_data) -t0_data/10)[:, np.newaxis]*data_sq).mean(axis=0)

            A4 = -1/10 * (mean_outer_product(z, z_sq) + mean_outer_product(data, data_sq))
            A5 = -1/10 * (mean_outer_product(z, z)    + mean_outer_product(data, data))
            A6 =  1/10 * z.mean(axis=0)                  + 1/10*data.mean(axis=0)
            b2 = -((deriv_sigmoid(t0_z)+t0_z/10)[:,np.newaxis]*z).mean(axis=0) +((deriv_sigmoid(t0_data) -t0_data/10)[:,np.newaxis]*data).mean(axis=0)

            A7 = -1/10 * z_sq.mean(axis=0) + 1/10 * data_sq.mean(axis=0)
            A8 = -1/10 * z.mean(axis=0)    + 1/10 * data.mean(axis=0)
            A9 = np.full(1, -0.2)
            b3 = np.array([np.mean(deriv_sigmoid(t0_z)  -t0_z/10, axis = 0) - np.mean(deriv_sigmoid(t0_data) -t0_data/10, axis=0)])
            
            A_a = np.concatenate([A1, A2, A3[:, np.newaxis]], axis=1)
            A_b = np.concatenate([A4, A5, A6[:, np.newaxis]], axis=1)
            A_bias = np.concatenate([A7, A8, A9], axis=0)
            A = np.concatenate([A_a, A_b, A_bias[np.newaxis, :]], axis=0)
            b = np.concatenate([b1, b2, b3], axis=0)
            decayed_lr_u =lr_u/j**dicay_par
            par = par*(1-lr_u) + lr_u*np.linalg.solve(A, b)
            l +=1
            # ここまで

            
        alpha_m = alpha[0]; alpha_v = alpha[1]
        mgrad = (z-alpha_m)
        sig_ = sigmoid(np.dot(np.stack([z**2, z], axis=1).reshape(m, 2*data_dim),par[0:2*data_dim ])- par[2*data_dim])[:,np.newaxis]
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
            exper_iter, optim_iter, L, init_loc, std,date, time, lr_u ]
    l = list(map(str, l))
    f.writelines(" ,".join(l))
