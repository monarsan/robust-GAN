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
import sys 
sys.path.append("../")
from libs.create import create_norm_data, create_sparse_cov
from libs.functions import sigmoid, g_lo, g_up
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
learn_par_u = float(argv[14])
init_loc=float(argv[13])
par_cov, out_cov = np.eye(data_dim), np.eye(data_dim)


# 分散を単位行列で固定
# 分散を単位行列で固定
res_mean = [0 for i in range(exper_iter)]
res_par = [0 for i in range(exper_iter)]
for i in range(exper_iter):
    print("%d/%d" %(i+1, exper_iter))
    data = create_norm_data(n, eps, par_mu, par_cov, out_mu, out_cov)
    mean_hist = []
    par_hist = []
    # 平均は次元ごとにロバスト、分散はロバストでない
    alpha = [np.median(data, axis=0), par_cov]
    z = np.random.multivariate_normal(mean=alpha[0], cov=alpha[1], size = m)

    def discriminator(x, beta):
        size, data_dim = x.shape[0], x.shape[-1]
        stack = [x]
        value = np.dot(np.stack(stack, axis=1).reshape(size, len(stack)*data_dim),beta)
        return value

    size_par = data_dim
    #par = np.random.normal(loc = 0, scale = 0.1, size = size_par)
    par = alpha[0]
    bias = np.array(np.mean( discriminator(z, par[0:size_par]) ))[np.newaxis]
    par = np.concatenate([par, bias], axis = 0)
    for j in tqdm(range(1, optim_iter+1)):
        z = np.random.multivariate_normal(mean=alpha[0], cov=alpha[1] , size = m)
        def major_func(par, past_par):
            new_beta = par[0:size_par]; new_b = par[size_par]; beta = past_par[0:size_par]; b = past_par[size_par]
            A = np.mean(g_lo(discriminator(z, new_beta) - new_b, discriminator(z, beta) - b))
            B = np.mean(g_up(discriminator(data, new_beta) - new_b, discriminator(data, beta) - b))
            reg = LA.norm(new_beta, ord=2)*par_reg1
            return -(A-B - reg)

        l = 0
        while(l<L):
            op = minimize(major_func, x0 = par, args = par)
            decaied_lr = learn_par_u/j**dicay_par
            par = op.x *decaied_lr + par*(1-decaied_lr)
            l+=1
        
        alpha_m = alpha[0]; alpha_v = alpha[1]
        mgrad = ((z-alpha_m)[:, np.newaxis, :]).sum(axis=2)
        sig_ = sigmoid(discriminator(z, par[:size_par]) - par[size_par])[:,np.newaxis]
        tmp_alpha_m = alpha[0] - learn_par/j**dicay_par * np.mean(mgrad*sig_, axis = 0)
        alpha[0] = tmp_alpha_m

        mean_hist.append(alpha[0])
        par_hist.append(par)
    res_mean[i] = mean_hist
    res_par[i] = par_hist


import os
file_name=0
while True:
    path = "result/res"+str(file_name)+".npy"
    if not os.path.isfile(path):
        break
    file_name+=1


np.save(path,np.array(res_mean))
path = "result/par"+str(file_name)+".npy"
np.save(path,np.array(res_par))

mean = []; npmu = np.array(res_mean)
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
            exper_iter, optim_iter, L, init_loc, std,date, time, learn_par_u ]
    l = list(map(str, l))
    f.writelines(" ,".join(l))
