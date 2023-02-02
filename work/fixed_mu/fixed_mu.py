import sys
sys.path.append("../")
import numpy as np
import numpy.linalg as LA
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from numpy.lib.function_base import cov
from sys import argv
from libs.functions import sigmoid, g_lo, g_up, nearPD
from libs.create import create_sparse_cov, create_norm_data

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


# 分散を固定しない
out_cov = np.eye(data_dim) #create_out_cov(data_dim)
res_mean = [0 for i in range(exper_iter)]
res_cov = [0 for i in range(exper_iter)]
res_par = [0 for i in range(exper_iter)]
ux=[0 for i in range(exper_iter)]
for i in (range(exper_iter)):
    data = create_norm_data(n, eps, par_mu, par_sd, out_mu, out_cov)

    cov_hist = []
    par_hist = []
    ux_hist = []
    # 平均は次元ごとにロバスト、分散はロバストでない
    alpha = [np.full(data_dim, par_mu), np.cov(data, rowvar = False)]
    z = np.random.multivariate_normal(mean=alpha[0], cov=alpha[1], size = m)

    def discriminator(x, beta):
        size, data_dim = x.shape[0], x.shape[-1]
        stack = [x, x**2]
        value = np.dot(np.stack(stack, axis=1).reshape(size, len(stack)*data_dim),beta)
        return value
    size_par = 2*data_dim
    par = np.random.normal(loc = 0, scale = 0.1, size = size_par)
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
            ux_hist.append(discriminator(z,par[:-1]).mean(axis=0) - discriminator(data,par[:-1]).mean(axis=0))
            op = minimize(major_func, x0 = par, args = par,  method=optim_method)
            par = op.x
            l+=1
        
        alpha_m = alpha[0]; alpha_v = alpha[1]
        v_inv = np.linalg.inv(alpha_v)
        sigma_grad = (alpha_v - (z- alpha_m)[:,:,np.newaxis] * (z-alpha_m)[:, np.newaxis, :])/2
        sig_ = sigmoid(discriminator(z, par[:2*data_dim]) - par[2*data_dim])[:,np.newaxis]
        tmp_alpha_v = v_inv - learn_par/j**dicay_par * np.mean(sigma_grad*sig_[:,:,np.newaxis], axis = 0)
        alpha[1] = (LA.inv(tmp_alpha_v))


        cov_hist.append(alpha[1])
        par_hist.append(par)
    res_cov[i] = cov_hist
    res_par[i] = par_hist
    ux[i] = ux_hist


import os
file_name=0
while True:
    path = "result/cov"+str(file_name)+".npy"
    if not os.path.isfile(path):
        break
    file_name+=1


path = "result/cov"+str(file_name)+".npy"
np.save(path,np.array(res_cov))
path = "result/par"+str(file_name)+".npy"
np.save(path,np.array(res_par))
path = "result/ux"+str(file_name)+".npy"
np.save(path,np.array(ux))

cov = []; npcov=np.array(res_cov)

    
for i in range(exper_iter):
    loss_cov = LA.norm(npcov[i]-np.eye(data_dim), axis=(1,2))
    cov.append(loss_cov[-1])

average_loss_cov = str(np.mean(cov))[:6]


import datetime
date = str(datetime.datetime.now())[:-10][:10]
time = str(datetime.datetime.now())[:-10][10:]

with open("./exper_result.csv", mode="a") as f:
    f.write("\n")
    l =  [file_name, average_loss_cov ,n, eps, data_dim,
            mu, mu_out,
            par_reg1, learn_par, dicay_par,
            exper_iter, optim_iter, L, init_loc,date, time]
    l = list(map(str, l))
    f.writelines(" ,".join(l))